from typing import Callable, Dict, List, Literal, Tuple
import torch
from torch.nn import Module
from torch import Tensor
from torch import vmap
from torch.nn import ModuleList
import ot
from loguru import logger
import typer


def _l2_metric(x: Tensor, y: Tensor) -> Tensor:
    return torch.dist(x, y, p=2)


def optimal_transport(xd: Tensor, yd: Tensor, mask: Tensor) -> Tensor:
    costs = torch.cdist(xd, yd, p=2)
    costs[mask <= 0] = torch.inf
    a = torch.ones(xd.shape[1], dtype=torch.float32, device=xd.device) / xd.shape[0]
    b = torch.ones(yd.shape[1], dtype=torch.float32, device=xd.device) / yd.shape[0]
    gs = []
    for i in range(xd.shape[0]):
        c = costs[i]
        g = ot.emd(a, b, c, numThreads='max')
        gs.append(g)
    gs = torch.stack(gs, dim=0)
    gs = gs / gs.amax(dim=(-1, -2), keepdim=True)
    return gs


def round_nn(xd: Tensor, yd: Tensor, mask: Tensor, ratio_thr: float = 0.6, max_matched_ratio: float = 0.01) -> Tensor:
    dist = torch.cdist(xd, yd, p=2)
    dist[mask <= 0] = torch.inf

    thr = torch.nanquantile(dist.flatten(-2), max_matched_ratio, dim=1, keepdim=True)
    valid = dist <= thr.unsqueeze(-1)
    row_ind = torch.argmin(dist, dim=-1, keepdim=True) == torch.arange(dist.shape[-1], device=dist.device).view(
        1, 1, -1
    )
    row_top2 = torch.topk(dist, k=2, dim=-1, largest=False).values
    row_ratio_test = (row_top2[..., 0] / (1e-6 + row_top2[..., 1])) < ratio_thr

    row_ind = row_ind & row_ratio_test.unsqueeze(-1)
    col_ind = torch.argmin(dist, dim=-2, keepdim=True) == torch.arange(dist.shape[-2], device=dist.device).view(
        1, -1, 1
    )
    col_top2 = torch.topk(dist, k=2, dim=-2, largest=False).values

    col_ratio_test = (col_top2[..., 0, :] / (1e-6 + col_top2[..., 1, :])) < ratio_thr
    col_ind = col_ind & col_ratio_test.unsqueeze(-2)
    # logger.debug(f'shapes of row_ind and col_ind: {row_ind.shape}, {col_ind.shape}')
    is_round = row_ind & col_ind
    # logger.debug(f'average round-way match: {is_round.sum(dim=(-1, -2)).float().mean()}')
    is_round: Tensor = is_round & (mask > 0) & valid
    return is_round


def nearest_neighbor(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    metric: Callable[[Tensor, Tensor], Tensor] | Module,
    orientation: Literal['x', 'y', 'xy'] = 'x',
    smallest: bool = True,
    topk: int | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Nearest neighbor matching.

    Args:
        x (Tensor): The first set of features with shape :math:`(B, N, D)`.
        y (Tensor): The second set of features with shape :math:`(B, M, D)`.
        mask (Tensor): The mask with shape :math:`(B, N, M)`.
        metric (Callable[[Tensor | float, Tensor | float], Tensor | float]): The metric function.
        orientation (Literal['x', 'y', 'xy'], optional): The orientation of the matching. Defaults to 'x'.
        smallest (bool, optional): Whether to find the smallest or largest distance. Defaults to True.
        topk (int | None, optional): The number of top matches to return. Defaults to None.

    Returns:
        Tensor: The matching mask with shape :math:`(B, N, M)`.

    """
    if orientation == 'xy' and topk is None:
        raise ValueError('`topk` must be specified when `orientation` is `xy`.')

    metric = vmap(metric, in_dims=(0, 0), out_dims=0)
    dists = metric(x, y)
    dists = dists * mask
    pairs: Tensor | None = None
    if orientation == 'x':
        top_dists, pairs = dists.min(dim=-1) if smallest else dists.max(dim=-1)
    elif orientation == 'y':
        top_dists, pairs = dists.min(dim=-2) if smallest else dists.max(dim=-2)
    elif orientation == 'xy':
        assert topk is not None
        dists = dists.flatten(-2)
        top_dists, top_idx = torch.topk(dists, topk, dim=-1, largest=not smallest)
        x_idx = top_idx // dists.shape[-1]
        y_idx = top_idx % dists.shape[-1]
        pairs = torch.stack([x_idx, y_idx], dim=-1)
    else:
        raise ValueError(f'Invalid orientation: {orientation}')

    # pairs in shape (B, _, 2)

    assert pairs is not None
    if topk is not None and top_dists.shape[-1] > topk:
        top_dists, top_pairs = torch.topk(top_dists, topk, dim=-1, largest=not smallest)
        pairs = pairs.gather(-1, top_pairs.unsqueeze(-1).expand(-1, -1, 2))

    mask = torch.zeros_like(mask)
    mask.scatter_(-1, pairs, 1)
    return mask, pairs, top_dists
