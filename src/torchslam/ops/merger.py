from typing import Tuple
import torch
from torch import Tensor
from .ba import proj
import math
import ot
import time
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
from torch_scatter import scatter
from loguru import logger
from . import ba
import cugraph
import pandas as pd
from .functional.cluster import kmeans


def merge_keyframes(
    x: Tensor, p: Tensor, d: Tensor, R: Tensor, t: Tensor, thr: float = 0.001
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dists = torch.cdist(x, x)
    to_merge = dists <= thr
    B, N, D = d.shape
    new_x = []
    new_p = []
    new_d = []
    new_R = []
    new_t = []
    merged = torch.zeros(x.size(0), device=x.device).bool()
    for idx, row in enumerate(to_merge.unbind()):
        row = row & ~merged
        if torch.sum(row) == 0:
            continue
        if torch.sum(row) == 1:
            new_x.append(x[row])
            new_p.append(p[row])
            new_d.append(d[row])
            new_R.append(R[row])
            new_t.append(t[row])
            continue
        row_x, row_p, row_d, row_R, row_t = merge_newframes(
            x[row].view(-1, 3),
            p[row].view(-1, N, 3),
            d[row].view(-1, N, D),
            R[row].view(-1, 3, 3),
            t[row].view(-1, 3),
        )
        new_x.append(row_x)
        new_p.append(row_p)
        new_d.append(row_d)
        new_R.append(row_R)
        new_t.append(row_t)
        merged = merged | row

    new_x = torch.cat(new_x)
    new_p = torch.cat(new_p)
    new_d = torch.cat(new_d)
    new_R = torch.cat(new_R)
    new_t = torch.cat(new_t)

    return new_x, new_p, new_d, new_R, new_t


def merge_newframes(x: Tensor, p: Tensor, d: Tensor, R: Tensor, t: Tensor) -> Tuple[Tensor, ...]:
    """
    Args:
        x: (B, 3)
        p: (B, N, 3)
        d: (B, N, D)
        R: (B, 3, 3)
        t: (B, 3)
        w: (B, 1)
    """
    B, N, D = d.shape
    K = int(N * 2)
    p = p
    p_world = proj(p, torch.linalg.pinv(R), -t)  # (B, N, 3)
    p_world = p_world.view(B * N, 3)
    cl, c = kmeans(p_world, k=K, its=32)  # (B * N,), (K, 3)

    d_c = torch.zeros(N, D, dtype=x.dtype, device=p.device)
    d_c = scatter(d.view(B * N, D), cl, dim=0, reduce='mean')
    pop = torch.bincount(cl, minlength=K).type_as(c)  # (K,)

    topN_vals, topN = torch.topk(pop, N, dim=0)  # (N,)

    if torch.any(topN_vals == 0):
        return x[-1], p[-1], d[-1], R[-1], t[-1]
    d_c = d_c[topN]
    c = c[topN]
    x = x[-1]
    R = R[-1]
    t = t[-1]
    kpts = proj(c.view(1, N, 3), R, t)  # (N, 3)
    kpts = kpts.view(1, N, 3)
    decs = d_c.view(1, N, D)
    R = R.view(1, 3, 3)
    t = t.view(1, 3)

    return x.view(1, 3), kpts, decs, R, t
