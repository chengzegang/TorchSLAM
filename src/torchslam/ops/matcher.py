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


def _batch_randperm(b: int, size: int, device: str | torch.device = 'cpu') -> Tensor:
    r"""Generate a batch of random permutations.

    Args:
        b (int): Batch size.
        size (int): Size of the permutations.
        device (str | torch.device, optional): Device to use. Defaults to 'cpu'.

    Returns:
        Tensor: A batch of random permutations.

    """

    perms = torch.argsort(torch.rand((b, size), device=device), dim=-1)
    return perms


def ransac(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    solver: Callable | Module,
    evaluator: Callable | Module,
    ransac_ratio: float = 0.6,
    ransac_it: int = 16,
    ransac_thr: float = 0.75,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""RANSAC algorithm to find the best model.

    Args:
        x (Tensor): The first set of features with shape :math:`(B, N, D)`.
        y (Tensor): The second set of features with shape :math:`(B, M, D)`.
        mask (Tensor): The mask with shape :math:`(B, N, M)`.
        solver (Callable): The solver function to find the model.
        evaluator (Callable): The evaluator function to evaluate the model.
        ransac_ratio (float, optional): The ratio of inliers to consider the model as the best. Defaults to 0.6.
        ransac_it (int, optional): The number of iterations. Defaults to 16.
        ransac_thr (float, optional): The threshold to consider a point as an inlier. Defaults to 0.75.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The matching mask with shape :math:`(B, N, M)`.
    """

    B, N, D = x.shape
    B, M, D = y.shape

    r_n = int(ransac_ratio * N)
    r_m = int(ransac_ratio * M)
    perm1 = _batch_randperm(ransac_it, r_n, device=x.device).view(-1)
    perm2 = _batch_randperm(ransac_it, r_m, device=x.device).view(-1)

    s_x = x[:, perm1].view(B * ransac_it, r_n, D)
    s_y = y[:, perm2].view(B * ransac_it, r_m, D)
    s_m = mask[:, perm1].view(B, ransac_it, r_n, M)
    s_m = s_m.gather(-1, perm2.view(1, ransac_it, 1, r_m).repeat(B, 1, r_n, 1)).view(
        B * ransac_it, r_n, r_m
    )  # (B * ransac_it, N, M)

    models = solver(s_x, s_y, s_m)  # (B * ransac_it, D, D)
    x = x.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    y = y.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    mask = mask.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    errors = evaluator(models, x, y, mask)  # (B * ransac_it, N, M)
    errors = errors.view(B, ransac_it, N, M)
    models = models.view(B, ransac_it, models.shape[-2], models.shape[-1])
    avg_errors = torch.nanmean(errors, dim=(-1, -2))
    best_model_idx = torch.argmin(avg_errors, dim=-1)

    best_model = torch.gather(
        models, dim=1, index=best_model_idx.view(-1, 1, 1, 1).repeat(1, 1, models.shape[-2], models.shape[-1])
    ).squeeze(1)

    best_errors = torch.gather(
        errors, dim=1, index=best_model_idx.view(-1, 1, 1, 1).repeat(1, 1, errors.shape[-2], errors.shape[-1])
    ).squeeze(1)
    thrs = torch.nanquantile(best_errors, ransac_thr, dim=-1, keepdim=True)
    inliers = best_errors < thrs
    return best_model, inliers, best_errors


def knn_ransac(
    x: Tensor,
    y: Tensor,
    mask: Tensor,
    solver: Callable | Module,
    evaluator: Callable | Module,
    k: int = 1,
    ransac_ratio: float = 0.6,
    ransac_it: int = 16,
    ransac_thr: float = 0.75,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""KNN-RANSAC algorithm to find the best model.

    Args:
        x (Tensor): The first set of features with shape :math:`(B, N, D)`.
        y (Tensor): The second set of features with shape :math:`(B, S, M, D)`.
        mask (Tensor): The mask with shape :math:`(B, N, M)`.
        solver (Callable): The solver function to find the model.
        evaluator (Callable): The evaluator function to evaluate the model.
        k (int, optional): The number of nearest neighbors per query. Defaults to 1.
        ransac_ratio (float, optional): The ratio of inliers to consider the model as the best. Defaults to 0.6.
        ransac_it (int, optional): The number of iterations. Defaults to 16.
        ransac_thr (float, optional): The threshold to consider a point as an inlier. Defaults to 0.75.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The best model, the inlier mask and the errors.
    """
    B, N, D = x.shape
    B, S, M, D = y.shape
    k = min(k, S)
    x = x.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, N, D)
    y = y.view(B * S, M, D)
    best_model, inliers, best_errors = ransac(x, y, mask, solver, evaluator, ransac_ratio, ransac_it, ransac_thr)
    best_model = best_model.view(B, S, D, D)
    inliers = inliers.view(B, S, N, M)
    best_errors = best_errors.view(B, S, N, M)
    avg_errors = torch.nanmean(best_errors, dim=(-2, -1))
    topk_errors, topk_idx = torch.topk(avg_errors, k, dim=-1)
    topk_models = torch.gather(best_model, dim=1, index=topk_idx.unsqueeze(-1).unsqueeze(-1).expand_as(best_model))
    topk_inliers = torch.gather(inliers, dim=1, index=topk_idx.unsqueeze(-1).unsqueeze(-1).expand_as(inliers))
    return topk_models, topk_inliers, topk_errors


class RANSACMatcher(Module):
    def __init__(
        self,
        solver: Callable | Module,
        evaluator: Callable | Module,
        ransac_ratio: float = 0.6,
        ransac_it: int = 16,
        ransac_thr: float = 0.75,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.solver = solver
        self.evaluator = evaluator
        self.ratio = ransac_ratio
        self.it = ransac_it
        self.thr = ransac_thr

    def forward(self, xk: Tensor, xd: Tensor, yk: Tensor, yd: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        w = round_nn(xd, yd, mask)
        best_model, inliers, best_errors = ransac(xk, yk, w, self.solver, self.evaluator, self.ratio, self.it, self.thr)
        return inliers, best_model, best_errors


class NNDFMatcher(Module):
    def forward(self, xd: Tensor, yd: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mask, pairs, top_dists = nearest_neighbor(xd, yd, mask, _l2_metric)
        return mask, pairs, top_dists


# K-Nearest Neighbor Key-Point Feature (KNNKPF) Node-level matcher
class KNNKPFNodeMatcher(Module):
    def __init__(
        self,
        solver: Callable | Module,
        evaluator: Callable | Module,
        k: int = 8,
        ransac_ratio: float = 0.6,
        ransac_it: int = 16,
        ransac_thr: float = 0.75,
        **kargs,
    ):
        self.solver = solver
        self.evaluator = evaluator
        self.k = k
        self.ransac_ratio = ransac_ratio
        self.ransac_it = ransac_it
        self.ransac_thr = ransac_thr

    def forward(self, qk: Tensor, qmask: Tensor, kks: Tensor, kmask: Tensor):
        topk_models, topk_inliers, topk_errors = knn_ransac(
            qk,
            kks,
            qmask & kmask,
            self.solver,
            self.evaluator,
            self.k,
            self.ransac_ratio,
            self.ransac_it,
            self.ransac_thr,
        )
        return topk_inliers, topk_models, topk_errors
