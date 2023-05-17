import torch
from torch import Tensor
from torch.nn import Module
from typing import Callable, Tuple
from .functional.nearest import round_nn
from .functional.ransac import ransac


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
