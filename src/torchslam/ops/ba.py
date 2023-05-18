from typing import Callable
import torch
from torch import Tensor
from torch.optim import SGD, Adam
from loguru import logger
import typer
from torch.nn import Module

from torchslam.ops.functional.convert import quat_to_mat
from ..utils import log
import roma
import torch.nn.functional as F
from .functional.proj import sreproj, reproj, proj
import ot


def _forward_pass(p: Tensor, phi: Tensor, t: Tensor) -> Tensor:
    # R = so3(phi)
    R = quat_to_mat(phi)
    ph = sreproj(p, R, t)
    return ph


def _expand_flat(x: Tensor, y: Tensor, mask: Tensor | None = None) -> tuple[Tensor, ...]:
    B1 = x.shape[0]
    B2 = y.shape[0]
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    x = x.repeat(1, B2, *([1] * (x.ndim - 2)))
    y = y.repeat(B1, 1, *([1] * (y.ndim - 2)))
    x = x.flatten(0, 1)
    y = y.flatten(0, 1)
    if mask is not None:
        m1 = mask.unsqueeze(1).unsqueeze(-1)
        m2 = mask.unsqueeze(0).unsqueeze(-2)
        mask = (m1 + m2) / 2
        mask = mask.flatten(0, 1)
        return x, y, mask
    return x, y


def bundle_adjust(
    p: Tensor, d: Tensor, mask: Tensor, matcher: Module | Callable, it: int = 1000
) -> tuple[Tensor, Tensor]:
    """
    p: (B, N, 3)
    d: (B, N, D)
    mask: (B, N)
    phi: (B, 3)
    t: (B, 3)
    """
    B, N, D = d.shape
    # log.shapes(p=p, d=d, mask=mask)
    p1, p2, m1 = _expand_flat(p, p, mask)
    d1, d2, m2 = _expand_flat(d, d, mask)
    m = (m1 + m2) > 0
    # log.shapes(p1=p1, p2=p2, d1=d1, d2=d2, m=m)
    ins, _, _ = matcher(p1, d1, p2, d2, m)
    ins = ins.view(B, B, N, N)

    # mask out batch-wise diagonal
    dig_mask = torch.eye(B, device=d.device, dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)
    ins.masked_fill_(dig_mask, 0)
    # mask out point-wise diagonal
    ins.diagonal(-1, -2).zero_()

    w = ins.type_as(d)
    phi = torch.randn(B, 4, device=d.device).requires_grad_(True)
    t = torch.randn(B, 3, device=d.device).requires_grad_(True)

    opt = Adam([phi, t], lr=1e-2)

    # p1 in B x 1 x N x 3
    # p2 in 1 x B x N x 3
    typer.echo('Bundle adjustment')

    with typer.progressbar(range(it)) as progress:
        for _ in progress:
            opt.zero_grad()
            ph = _forward_pass(p, phi, t)  # B x B x N x 3
            # p: B x N x 3
            # compare every two points among all frames, the most generalized form
            err = torch.norm(ph.unsqueeze(-2) - p.view(1, B, 1, N, 3), dim=-1, p=2)
            # batch size averaging
            err = (w * err).sum(dim=(-1, -2)) / ((torch.isfinite(w) & (w >= 1e-8)).sum(dim=(-1, -2)) + 1e-8)
            err = err.nanmean()
            err.backward()
            opt.step()
            progress.label = f'err: {err.item():.6f}'
            if err <= 0.002:
                break
    R = quat_to_mat(phi).detach()  # B x 3 x 3
    # R = so3(phi).detach()  # B x 3 x 3
    t = t.detach()  # B x 3
    return R, t
