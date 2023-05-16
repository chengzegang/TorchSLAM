from typing import Tuple
import torch
from torch import Tensor
from .pnp import proj
import math
import ot
import time
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
from torch_scatter import scatter
from loguru import logger

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"


def KMeans(x: Tensor, k: int = 10, its: int = 10) -> Tuple[Tensor, Tensor]:
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:k, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, k, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(its):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=k).type_as(c).view(k, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c


def merge(p: Tensor, d: Tensor, R: Tensor, t: Tensor) -> Tuple[Tensor, ...]:
    """
    Args:
        p: (B, N, 3)
        d: (B, N, D)
        R: (B, 3, 3)
        t: (B, 3)
    """
    B, N, D = d.shape
    K = N * 2
    p_world = proj(p, torch.linalg.pinv(R), -t)  # (B, N, 3)
    p_world = p_world.view(B * N, 3)
    cl, c = KMeans(p_world, k=K, its=32)  # (B * N,), (K, 3)
    # logger.debug(f"cl: {cl.shape}, c: {c.shape}")
    d_c = torch.zeros(N, D, dtype=dtype, device=p.device)
    d_c = scatter(d.view(B * N, D), cl, dim=0, reduce="mean")  # (N, D)
    # logger.debug(f"d_c: {d_c.shape}")
    pop = torch.bincount(cl, minlength=K).type_as(c)  # (K,)

    topN_vals, topN = torch.topk(pop, N, dim=0)  # (N,)

    # logger.debug(f"topN_vals: {topN_vals.shape}, topN: {topN.shape}")
    if torch.any(topN_vals == 0):
        return (
            torch.empty(0, N, 3, dtype=dtype, device=p.device),
            torch.empty(0, N, D, dtype=dtype, device=p.device),
            torch.empty(0, 3, 3, dtype=dtype, device=p.device),
            torch.empty(0, 3, dtype=dtype, device=p.device),
        )

    d_c = d_c[topN]
    c = c[topN]
    R = R[0]
    t = t[0]
    kpt = proj(c, R, t)  # (N, 3)
    kpt = kpt.view(1, N, 3)
    decs = d_c.view(1, N, D)
    R = R.view(1, 3, 3)
    t = t.view(1, 3)
    return kpt, decs, R, t
