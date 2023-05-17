from typing import Callable
import torch
from torch import Tensor
from torch.optim import Adam
from loguru import logger
import typer
from torch.nn import Module
from ..utils import log
import roma
import torch.nn.functional as F


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    q = F.normalize(q, dim=-1, p=2)
    rot: Tensor = roma.unitquat_to_rotmat(q)
    return rot


def so3(phi: Tensor) -> Tensor:
    B = phi.shape[0]
    R = torch.zeros(B, 3, 3, device=phi.device, dtype=phi.dtype)
    R[:, 0, 1] = -phi[:, 2]
    R[:, 0, 2] = phi[:, 1]
    R[:, 1, 0] = phi[:, 2]
    R[:, 1, 2] = -phi[:, 0]
    R[:, 2, 0] = -phi[:, 1]
    R[:, 2, 1] = phi[:, 0]
    R = R + torch.eye(3, device=phi.device, dtype=phi.dtype).view(1, 3, 3)
    return R


def proj(x: Tensor, R: Tensor, t: Tensor) -> Tensor:
    '''
    x: (*, N, 3)
    R: (*, 3, 3)
    t: (*, 3)
    '''
    x = (x @ R.transpose(-1, -2)) + t.unsqueeze(-2)
    return x


def reproj(x: Tensor, R1: Tensor, t1: Tensor, R2: Tensor, t2: Tensor) -> Tensor:
    x = proj(x, torch.linalg.pinv(R1), -t1)
    x = proj(x, R2, t2)
    return x


def creproj(x: Tensor, R1: Tensor, t1: Tensor, R2: Tensor, t2: Tensor) -> Tensor:
    '''
    x: (B1, N, 3)
    R1: (B1, 3, 3)
    t1: (B1, 3)
    R2: (B2, 3, 3)
    t2: (B2, 3)
    '''
    x = proj(x, torch.linalg.pinv(R1), -t1)  # (B, N, 3)
    x = x.unsqueeze(1)
    R2 = R2.unsqueeze(0)
    t2 = t2.unsqueeze(0)
    x = proj(x, R2, t2)  # (B, B, N, 3)
    return x


def sreproj(x: Tensor, R: Tensor, t: Tensor) -> Tensor:
    '''
    x: (B, N, 3)
    R: (B, 3, 3)
    t: (B, 3)
    '''
    B, N, _ = x.shape
    x = proj(x, torch.linalg.pinv(R), -t)  # (B, N, 3)
    x = x.unsqueeze(1)
    R = R.unsqueeze(0)
    t = t.unsqueeze(0)
    x = proj(x, R, t)  # (B, B, N, 3)
    return x


def _forward_pass(p: Tensor, phi: Tensor, t: Tensor) -> Tensor:
    # R = so3(phi)
    R = quaternion_to_rotation_matrix(phi)
    ph = sreproj(p, R, t)
    return ph


def _expand_flat(x: Tensor, y: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    B1 = x.shape[0]
    B2 = y.shape[0]
    x = x.unsqueeze(1)
    m1 = mask.unsqueeze(1).unsqueeze(-1)
    y = y.unsqueeze(0)
    m2 = mask.unsqueeze(0).unsqueeze(-2)
    x = x.repeat(1, B2, *([1] * (x.ndim - 2)))
    y = y.repeat(B1, 1, *([1] * (y.ndim - 2)))
    x = x.flatten(0, 1)
    y = y.flatten(0, 1)
    mask = m1 & m2
    mask = mask.flatten(0, 1)
    return x, y, mask


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
    m = m1 & m2
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

    # .shapes(p=p, d=d, mask=mask, phi=phi, t=t, ins=ins, w=w)

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
            progress.label = f'err: {err.item():.4f}'
            if err <= 0.001:
                break
    R = quaternion_to_rotation_matrix(phi).detach()  # B x 3 x 3
    # R = so3(phi).detach()  # B x 3 x 3
    t = t.detach()  # B x 3
    return R, t
