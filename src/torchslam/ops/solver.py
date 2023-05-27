from typing import Tuple
import torch
from torch import Tensor

from torch.nn import Module
import torch.nn.functional as F
from loguru import logger

from torchslam.ops.functional.solve import find_fundamental, find_fundamental_equirectangular
from .functional.convert import to_homogeneous

from .functional.proj import spherical


class Fundamental(Module):
    def __init__(self, camera_model: str = 'pinhole', **kwargs):
        super().__init__()
        self.camera_model = camera_model

    def forward(self, p1: Tensor, p2: Tensor, mask: Tensor) -> Tensor:
        r"""Estimates the fundamental matrix from a set of point correspondences.

        Args:
            p1: The set of points seen from the first camera frame in the camera plane
              coordinates with shape :math:`(*, N, 2)`.
            p2: The set of points seen from the second camera frame in the camera plane
              coordinates with shape :math:`(*, N, 2)`.
            mask: The mask of valid points with shape :math:`(*, N)`.

        Returns:
            The estimated fundamental matrix with shape :math:`(*, 3, 3)`.
        """
        if self.camera_model == 'pinhole':
            return find_fundamental(p1, p2, mask)
        elif self.camera_model == 'equirectangular':
            return find_fundamental_equirectangular(p1, p2, mask)
        else:
            raise NotImplementedError(self.camera_model)
