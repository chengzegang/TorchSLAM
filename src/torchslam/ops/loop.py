from typing import Tuple
import torch
from torch import Tensor


def loop_closure(
    curr_locs: Tensor,
    curr_kpts: Tensor,
    curr_descs: Tensor,
    keyframe_locs: Tensor,
    keyframe_kpts: Tensor,
    keyframe_descs: Tensor,
) -> Tuple[Tensor, ...]:
    return NotImplemented
