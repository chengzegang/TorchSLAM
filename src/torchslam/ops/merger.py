from typing import Tuple
import torch
from torch import Tensor

from torch_scatter import scatter

from .functional.proj import proj
from .functional.cluster import kmeans
import torch.nn.functional as F


merged_desc_avg_dist = 0.0
lr = 0.99
decay = 0.9


def merge_keyframes(
    x: Tensor, p: Tensor, d: Tensor, R: Tensor, t: Tensor, thr: float = 0.0001
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


def merge_newframes(x: Tensor, p: Tensor, d: Tensor, R: Tensor, t: Tensor, K: int | None = None) -> Tuple[Tensor, ...]:
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
    K = int(N * 2) if K is None else K
    p = p
    p_world = proj(p, torch.linalg.pinv(R), -t)  # (B, N, 3)
    p_world = p_world.view(B * N, 3)
    cl, c = kmeans(p_world, k=K, its=32)  # (B * N,), (K, 3)

    d_c = torch.zeros(N, D, dtype=x.dtype, device=p.device)
    d_c = scatter(d.reshape(B * N, D), cl, dim=0, reduce='mean')
    diff = torch.norm(d.view(B * N, D) - d_c[cl], dim=-1, p=2)
    global merged_desc_avg_dist, lr, decay
    merged_desc_avg_dist = merged_desc_avg_dist * (1 - lr) + torch.median(diff).item() * lr
    lr = lr * decay
    pop = torch.bincount(cl, minlength=K).type_as(c)  # (K,)

    topN_vals, topN = torch.topk(pop, N, dim=0)  # (N,)

    if torch.any(topN_vals == 0):
        return x[-1], p[-1], d[-1], R[-1], t[-1]
    d_c = d_c[topN]
    c = c[topN]
    R = R[-1]
    t = t[-1]
    x = x[-1]
    kpts = proj(c.view(1, N, 3), R, t)  # (N, 3)
    kpts = kpts.view(1, N, 3)
    descs = d_c.view(1, N, D)
    descs = F.normalize(descs, dim=-1, p=2)
    R = R.view(1, 3, 3)
    t = t.view(1, 3)

    return x.view(1, 3), kpts, descs, R, t


def merge_pairs(
    x: Tensor, xk: Tensor, xd: Tensor, xR: Tensor, xt: Tensor, y: Tensor, yk: Tensor, yd: Tensor, yR: Tensor, yt: Tensor
) -> Tuple[Tensor, ...]:
    """
    Args:
        x: (B, 3)
        p: (B, N, 3)
        d: (B, N, D)
        R: (B, 3, 3)
        t: (B, 3)
        w: (B, 1)
    """
    B, N, D = xd.shape

    ls = []
    ks = []
    ds = []
    Rs = []
    ts = []
    for i in range(B):
        loc = torch.stack([x[i], y[i]], dim=0)
        kpt = torch.stack([xk[i], yk[i]], dim=0)
        desc = torch.stack([xd[i], yd[i]], dim=0)
        R = torch.stack([xR[i], yR[i]], dim=0)
        t = torch.stack([xt[i], yt[i]], dim=0)

        l, k, d, R, t = merge_newframes(loc, kpt, desc, R, t)
        ls.append(l)
        ks.append(k)
        ds.append(d)
        Rs.append(R)
        ts.append(t)

    ls = torch.cat(ls)
    ks = torch.cat(ks)
    ds = torch.cat(ds)
    Rs = torch.cat(Rs)
    ts = torch.cat(ts)
    return ls.view(-1, 3), ks, ds, Rs, ts
