from queue import Full, Queue
import time
from typing import Callable, Tuple
import torch
from torch import Tensor
from . import ba
from . import merger
from .functional.proj import proj, reproj, creproj, sreproj
from torch_scatter import scatter
from torch.nn import Module, Parameter


def expoential_smoothing(x: Tensor, alpha: float) -> Tensor:
    factors = torch.full_like(x, alpha, dtype=x.dtype, device=x.device)
    factors = torch.cumprod(factors, dim=0).flip(0)
    weigted = x * factors
    res = torch.sum(weigted, dim=0) / torch.sum(factors, dim=0)
    return res


class LocalGraph(Module):
    def __init__(
        self,
        *,
        matcher: Module | Callable,
        feature_dim: int,
        num_features: int,
        world_dist_thr: float = 0.01,
        track_at_most_n_keyframes: int = 20,
        msg_queue: Queue | None = None,
        **kwargs,
    ):
        super().__init__()
        self.msg_queue = msg_queue
        self.matcher = matcher
        self._keyframe_locs = torch.empty(0, 3)
        self._keyframe_kpts = torch.empty(0, num_features, 3)
        self._keyframe_descs = torch.empty(0, num_features, feature_dim)
        self._keyframe_Rs = torch.empty(0, 3, 3)
        self._keyframe_ts = torch.empty(0, 3)
        self._track = torch.empty(0, 3)
        self.world_dist_thr = world_dist_thr
        self.track_at_most_n_keyframes = track_at_most_n_keyframes

    @property
    def keyframe_locs(self) -> Tensor:
        return self._keyframe_locs

    @property
    def keyframe_kpts(self) -> Tensor:
        return self._keyframe_kpts

    @property
    def keyframe_descs(self) -> Tensor:
        return self._keyframe_descs

    @property
    def keyframe_Rs(self) -> Tensor:
        return self._keyframe_Rs

    @property
    def keyframe_ts(self) -> Tensor:
        return self._keyframe_ts

    @property
    def keyframes(self) -> Tuple[Tensor, ...]:
        return (
            self.keyframe_locs,
            self.keyframe_kpts,
            self.keyframe_descs,
            self.keyframe_Rs,
            self.keyframe_ts,
        )

    @keyframes.setter
    def keyframes(self, t: Tuple[Tensor, ...]):
        self._keyframe_locs = t[0]
        self._keyframe_kpts = t[1]
        self._keyframe_descs = t[2]
        self._keyframe_Rs = t[3]
        self._keyframe_ts = t[4]

    def _extend_keyframes(self, new_locs: Tensor, new_kpts: Tensor, new_descs: Tensor, new_R: Tensor, new_t: Tensor):
        new_locs = new_locs.detach().cpu()
        new_kpts = new_kpts.detach().cpu()
        new_descs = new_descs.detach().cpu()
        new_R = new_R.detach().cpu()
        new_t = new_t.detach().cpu()

        self._keyframe_locs = torch.cat([self.keyframe_locs, new_locs], dim=0)
        self._keyframe_kpts = torch.cat([self.keyframe_kpts, new_kpts], dim=0)
        self._keyframe_descs = torch.cat([self.keyframe_descs, new_descs], dim=0)
        self._keyframe_Rs = torch.cat([self._keyframe_Rs, new_R], dim=0)
        self._keyframe_ts = torch.cat([self._keyframe_ts, new_t], dim=0)

        msg = (new_locs, new_kpts, new_descs, new_R, new_t)
        while self.msg_queue is not None:
            try:
                self.msg_queue.put(msg, timeout=1)
                break
            except Full:
                time.sleep(3)

    def _get_bundle(
        self, newframe_kpts: Tensor, newframe_descs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
        curr_kpts = newframe_kpts
        curr_descs = newframe_descs
        kf_locs: Tensor | None = None
        if torch.numel(self.keyframe_kpts) > 0:
            n_keyframe_used = min(self.keyframe_kpts.shape[0], self.track_at_most_n_keyframes)

            last_loc = self.track[-1].to(self.keyframe_locs.device)
            dists = torch.cdist(self.keyframe_locs, last_loc.view(1, 3)).view(-1)
            _, topk = torch.topk(dists, n_keyframe_used // 3, largest=False, sorted=True)

            kf_close_locs = self.keyframe_locs[topk].type_as(newframe_descs)
            kf_close_kpts = self.keyframe_kpts[topk].type_as(newframe_descs)
            kf_close_descs = self.keyframe_descs[topk].type_as(newframe_descs)

            kf_temp_locs = self.keyframe_locs[-n_keyframe_used // 3 :].type_as(newframe_descs)
            kf_temp_kpts = self.keyframe_kpts[-n_keyframe_used // 3 :].type_as(newframe_descs)
            kf_temp_descs = self.keyframe_descs[-n_keyframe_used // 3 :].type_as(newframe_descs)

            kf_locs = torch.cat([kf_temp_locs, kf_close_locs])
            kf_kpts = torch.cat([kf_temp_kpts, kf_close_kpts])
            kf_descs = torch.cat([kf_temp_descs, kf_close_descs])

            n_keyframe_used = kf_kpts.shape[0]
            curr_kpts = torch.cat([kf_kpts, curr_kpts]).type_as(newframe_descs)
            curr_descs = torch.cat([kf_descs, curr_descs]).type_as(newframe_descs)

        curr_mask = (
            torch.isfinite(curr_descs).all(dim=-1)
            & torch.isfinite(curr_kpts).all(dim=-1)
            & (curr_kpts.norm(dim=-1, p=2) >= 1e-6)
            & (curr_descs.norm(dim=-1, p=2) >= 1e-6)
        ).to(newframe_descs.device)
        return curr_kpts, curr_descs, curr_mask, kf_locs

    def _bundle_adjust(self, curr_kpts: Tensor, curr_descs: Tensor, curr_mask: Tensor) -> Tuple[Tensor, ...]:
        curr_R, curr_t = ba.bundle_adjust(curr_kpts, curr_descs, curr_mask, self.matcher)
        return curr_R, curr_t

    def _unpack(self, bundle_R: Tensor, bundle_t: Tensor, n_new: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        prev_R = bundle_R[:-n_new]
        prev_t = bundle_t[:-n_new]
        curr_R = bundle_R[-n_new:]
        curr_t = bundle_t[-n_new:]
        return curr_R, curr_t, prev_R, prev_t

    def _estimate(
        self,
        curr_R: Tensor,
        curr_t: Tensor,
        prev_R: Tensor | None = None,
        prev_t: Tensor | None = None,
        prev_locs: Tensor | None = None,
    ):
        if prev_locs is not None and prev_R is not None and prev_t is not None:
            newframe_locs = creproj(prev_locs.view(-1, 1, 3), prev_R, prev_t, curr_R, curr_t).squeeze(-2).mean(0)
        else:
            R1 = curr_R[:1]
            R2 = curr_R[1:]
            t1 = curr_t[:1]
            t2 = curr_t[1:]
            init_loc = torch.zeros(1, 1, 3, device=curr_R.device, dtype=curr_R.dtype)
            newframe_locs = reproj(init_loc, R1, t1, R2, t2).squeeze(-2)
            newframe_locs = torch.cat([init_loc.view(-1, 3), newframe_locs], dim=0)
        assert newframe_locs is not None
        return newframe_locs

    def _extend_track(self, locs: Tensor):
        self._track = torch.cat([self._track, locs.cpu()], dim=0)

    @property
    def track(self) -> torch.Tensor:
        return self._track

    def update(self, newframe_kpts: Tensor, newframe_descs: Tensor):
        bundle_kpts, bundle_descs, bundle_mask, keyframe_locs = self._get_bundle(newframe_kpts, newframe_descs)
        bundle_R, bundle_t = self._bundle_adjust(bundle_kpts, bundle_descs, bundle_mask)
        newframe_R, newframe_t, keyframe_R, keyframe_t = self._unpack(bundle_R, bundle_t, newframe_kpts.size(0))
        newframe_locs = self._estimate(newframe_R, newframe_t, keyframe_R, keyframe_t, keyframe_locs)

        self._extend_keyframes(
            *merger.merge_newframes(newframe_locs, newframe_kpts, newframe_descs, newframe_R, newframe_t)
        )
        self._extend_track(newframe_locs)

    def locate(self, kpts: Tensor, descs: Tensor) -> Tensor:
        return NotImplemented
