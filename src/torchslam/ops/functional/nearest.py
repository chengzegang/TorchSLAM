from typing import Callable, Literal, Tuple
import torch
from torch.nn import Module
from torch import Tensor
from torch import vmap
import ot


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
