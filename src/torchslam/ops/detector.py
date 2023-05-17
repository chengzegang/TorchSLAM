from __future__ import annotations

from typing import Tuple


import torch
from torch.nn import Module
from kornia.color import rgb_to_grayscale
from kornia.feature import (
    BlobHessian,
    BlobDoG,
    LAFOrienter,
    ScaleSpaceDetector,
    SIFTDescriptor,
    extract_patches_from_pyramid,
    get_laf_center,
)
from kornia.geometry import ConvQuadInterp3d, ScalePyramid
from torch import nn  # type: ignore
from tqdm import tqdm

from torch import Tensor


class SIFT(Module):
    def __init__(
        self,
        num_features: int = 512,
        patch_size: int = 41,
        angle_bins: int = 8,
        spatial_bins: int = 8,
        scale_n_levels: int = 3,
        sigma: float = 1.6,
        root_sift: bool = True,
        double_image: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.descriptor = SIFTDescriptor(patch_size, angle_bins, spatial_bins, rootsift=root_sift)
        self.detector = ScaleSpaceDetector(
            num_features,
            resp_module=BlobDoG(),
            scale_space_response=True,  # We need that, because DoG operates on scale-space
            nms_module=ConvQuadInterp3d(10),
            scale_pyr_module=ScalePyramid(scale_n_levels, sigma, patch_size, double_image=double_image),
            ori_module=LAFOrienter(19),
            mr_size=6.0,
            minima_are_also_good=True,
        )

    def detect(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self.detector.to(x.device)
        with torch.no_grad():
            lafs, resps = self.detector(x.contiguous())
            return lafs, resps

    def describe(self, x: Tensor, lafs: Tensor) -> Tensor:
        self.descriptor.to(x.device)
        with torch.no_grad():
            patches = extract_patches_from_pyramid(x, lafs, self.patch_size)
            B, N, CH, H, W = patches.size()
            descs = self.descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
            return descs  # type: ignore

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, C, H, W = x.shape
        x = rgb_to_grayscale(x).float()
        lafs, resps = self.detect(x)
        descs = self.describe(x, lafs)
        kpts = get_laf_center(lafs)
        x = kpts[..., 0]
        y = kpts[..., 1]
        x = x / W * 2 - 1
        y = y / H * 2 - 1
        kpts = torch.stack([x, y], dim=-1)
        return kpts, descs, resps
