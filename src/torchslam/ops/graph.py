import math
import os
from queue import Queue
from typing import Any, Callable, Tuple
import torch
from torch import Tensor

from torchslam.ops.database.cozo import CozoDB
from ..utils import config
from . import ba
from . import merger
from .functional.proj import reproj, creproj
from torch.nn import Module
from collections import deque
from torch_geometric.nn import knn
import typer


class Keyframes(Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._db = CozoDB(**kwargs)
        self._offline_keyframes: deque[Tuple[Tensor, ...]] = deque()

        _keyframe_locs: Tensor | None = None
        _keyframe_kpts: Tensor | None = None
        _keyframe_descs: Tensor | None = None
        _keyframe_Rs: Tensor | None = None
        _keyframe_ts: Tensor | None = None

        self.register_buffer('_keyframe_locs', _keyframe_locs)
        self.register_buffer('_keyframe_kpts', _keyframe_kpts)
        self.register_buffer('_keyframe_descs', _keyframe_descs)
        self.register_buffer('_keyframe_Rs', _keyframe_Rs)
        self.register_buffer('_keyframe_ts', _keyframe_ts)

    def __len__(self):
        return len(self.keyframe_locs) if self.keyframe_locs is not None else 0

    def offline(self, idx: Tensor):
        offline = (
            self.keyframe_locs[idx].detach().cpu(),
            self.keyframe_kpts[idx].detach().cpu(),
            self.keyframe_descs[idx].detach().cpu(),
            self.keyframe_Rs[idx].detach().cpu(),
            self.keyframe_ts[idx].detach().cpu(),
        )
        self._offline_keyframes.append(offline)

    @property
    def keyframe_locs(self) -> Tensor:
        if self._keyframe_locs is not None:
            return self._keyframe_locs
        else:
            return torch.empty(0, 3)

    @keyframe_locs.setter
    def keyframe_locs(self, loc: Tensor):
        self._keyframe_locs = loc

    @property
    def keyframe_kpts(self) -> Tensor:
        if self._keyframe_kpts is not None:
            return self._keyframe_kpts
        else:
            return torch.empty(0, config.num_features, 3)

    @keyframe_kpts.setter
    def keyframe_kpts(self, kpt: Tensor):
        self._keyframe_kpts = kpt

    @property
    def keyframe_descs(self) -> Tensor:
        if self._keyframe_descs is not None:
            return self._keyframe_descs
        else:
            return torch.empty(0, config.num_features, config.feature_dim)

    @keyframe_descs.setter
    def keyframe_descs(self, descs: Tensor):
        self._keyframe_descs = descs

    @property
    def keyframe_Rs(self) -> Tensor:
        return self._keyframe_Rs if self._keyframe_Rs is not None else torch.empty(0, 3, 3)

    @keyframe_Rs.setter
    def keyframe_Rs(self, R: Tensor):
        self._keyframe_Rs = R

    @property
    def keyframe_ts(self) -> Tensor:
        return self._keyframe_ts if self._keyframe_ts is not None else torch.empty(0, 3)

    @keyframe_ts.setter
    def keyframe_ts(self, t: Tensor):
        self._keyframe_ts = t

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

    def merge(self):
        if len(self) <= config.min_tracking_keyframes:
            return
        rand_idx = torch.randperm(len(self) - 3)[:8]
        rand_idx = torch.sort(rand_idx)[0]
        not_in = torch.ones(len(self), dtype=torch.bool)
        not_in = not_in.scatter_(0, rand_idx, False)

        new_kf_locs, new_kf_kpts, new_kf_descs, new_kf_R, new_kf_t = merger.merge_keyframes(
            self.keyframe_locs[rand_idx],
            self.keyframe_kpts[rand_idx],
            self.keyframe_descs[rand_idx],
            self.keyframe_Rs[rand_idx],
            self.keyframe_ts[rand_idx],
        )
        self.offline(rand_idx)
        self.keyframe_locs = self.keyframe_locs[not_in]
        self.keyframe_kpts = self.keyframe_kpts[not_in]
        self.keyframe_descs = self.keyframe_descs[not_in]
        self.keyframe_Rs = self.keyframe_Rs[not_in]
        self.keyframe_ts = self.keyframe_ts[not_in]

        closest = torch.argmin(torch.cdist(new_kf_locs, self.keyframe_locs), dim=-1)

        for idx, cls_idx in enumerate(closest):
            self.keyframe_locs = torch.cat(
                [self.keyframe_locs[:cls_idx], new_kf_locs[[idx]], self.keyframe_locs[cls_idx + 1 :]]
            )
            self.keyframe_kpts = torch.cat(
                [self.keyframe_kpts[:cls_idx], new_kf_kpts[[idx]], self.keyframe_kpts[cls_idx + 1 :]]
            )
            self.keyframe_descs = torch.cat(
                [self.keyframe_descs[:cls_idx], new_kf_descs[[idx]], self.keyframe_descs[cls_idx + 1 :]]
            )
            self.keyframe_Rs = torch.cat([self.keyframe_Rs[:cls_idx], new_kf_R[[idx]], self.keyframe_Rs[cls_idx + 1 :]])
            self.keyframe_ts = torch.cat([self.keyframe_ts[:cls_idx], new_kf_t[[idx]], self.keyframe_ts[cls_idx + 1 :]])

    def offload(self):
        os.makedirs('kfs', exist_ok=True)
        while len(self._offline_keyframes) > 0:
            torch.save(self._offline_keyframes.pop(), f'kfs/kf_{len(os.listdir("kfs"))}.pt')
        # while len(self._offline_keyframes) > 0:
        #    keyframes = self._offline_keyframes.pop()
        #    self._db.insert_keyframes(keyframes)

    def remove(self, idx: Tensor | None = None, mask: Tensor | None = None):
        if idx is not None:
            index = torch.arange(len(self))
            ins = torch.isin(index, idx).any(-1)
            index = index[~ins].view(-1)
            self.keyframe_locs = self.keyframe_locs[index]
            self.keyframe_kpts = self.keyframe_kpts[index]
            self.keyframe_descs = self.keyframe_descs[index]
            self.keyframe_Rs = self.keyframe_Rs[index]
            self.keyframe_ts = self.keyframe_ts[index]
        if mask is not None:
            self.keyframe_locs = self.keyframe_locs[~mask]
            self.keyframe_kpts = self.keyframe_kpts[~mask]
            self.keyframe_descs = self.keyframe_descs[~mask]
            self.keyframe_Rs = self.keyframe_Rs[~mask]
            self.keyframe_ts = self.keyframe_ts[~mask]

    def local_adjust(self):
        if len(self) <= config.min_tracking_keyframes:
            return
        rand_idx = torch.randperm(len(self) - 3)[:8]
        self.keyframe_locs, outs = ba.location_bundle_adjust(
            self.keyframe_locs[rand_idx],
            self.keyframe_kpts[rand_idx],
            self.keyframe_descs[rand_idx],
            self.keyframe_Rs[rand_idx],
            self.keyframe_ts[rand_idx],
        )
        self.remove(rand_idx[outs])

    def loc_knn(self, locs: Tensor, k: int = 1) -> Tuple[Tensor, Tensor]:
        if len(self.keyframe_locs) == 0:
            return torch.empty(0, dtype=torch.long, device=locs.device), torch.empty(
                0, dtype=torch.long, device=locs.device
            )

        kf_idxs = knn(
            locs.view(-1, 3), self.keyframe_locs.view(-1, 3).type_as(locs), k=k, num_workers=config.num_workers
        )
        dists = torch.dist(locs, self.keyframe_locs.to(locs.device)[kf_idxs])
        valid = dists <= config.map_resolution
        i = torch.arange(len(locs), dtype=torch.long).type_as(kf_idxs)
        i = i[valid]
        kf_idxs = kf_idxs[valid]
        return i, kf_idxs

    def desc_knn(self, descs: Tensor, k: int = 1) -> Tuple[Tensor, Tensor]:
        if len(self.keyframe_descs) == 0:
            return torch.empty(0, dtype=torch.long, device=descs.device), torch.empty(
                0, dtype=torch.long, device=descs.device
            )
        kf_idxs = knn(descs, self.keyframe_descs.type_as(descs), k=k, num_workers=config.num_workers)
        dists = torch.dist(descs, self.keyframe_descs.to(descs.device)[kf_idxs])
        valid = dists <= merger.merged_desc_avg_dist + 1e-8
        i = torch.arange(len(descs), dtype=torch.long, device=descs.device)
        i = i[valid]
        kf_idxs = kf_idxs[valid]
        return i, kf_idxs

    def update(self, new_locs: Tensor, new_kpts: Tensor, new_descs: Tensor, new_R: Tensor, new_t: Tensor):
        new_locs = new_locs.detach().cpu()
        new_kpts = new_kpts.detach().cpu()
        new_descs = new_descs.detach().cpu()
        new_R = new_R.detach().cpu()
        new_t = new_t.detach().cpu()
        if len(self) > 0 and len(self) > config.min_tracking_keyframes:
            i, j = self.loc_knn(new_locs.view(-1, 3))
            if torch.numel(i) > 0:
                i = torch.as_tensor(i, dtype=torch.long, device=new_locs.device)
                j = torch.as_tensor(j, dtype=torch.long, device=new_locs.device)
                l1 = new_locs[i]
                p1 = new_kpts[i]
                d1 = new_descs[i]
                R1 = new_R[i]
                t1 = new_t[i]

                l2 = self.keyframe_locs[j]
                p2 = self.keyframe_kpts[j]
                d2 = self.keyframe_descs[j]
                R2 = self.keyframe_Rs[j]
                t2 = self.keyframe_ts[j]

                l, k, d, R, t = merger.merge_pairs(l1, p1, d1, R1, t1, l2, p2, d2, R2, t2)
                self.keyframe_locs[j] = l
                self.keyframe_kpts[j] = k
                self.keyframe_descs[j] = d
                self.keyframe_Rs[j] = R
                self.keyframe_ts[j] = t

                index = torch.arange(len(new_kpts), device=new_kpts.device)
                isin = torch.isin(index, i).any(dim=1)
                index = index[~isin]
                new_locs = new_locs[index]
                new_kpts = new_kpts[index]
                new_descs = new_descs[index]
                new_R = new_R[index]
                new_t = new_t[index]

        self.keyframe_locs = torch.cat([self.keyframe_locs, new_locs], dim=0)
        self.keyframe_kpts = torch.cat([self.keyframe_kpts, new_kpts], dim=0)
        self.keyframe_descs = torch.cat([self.keyframe_descs, new_descs], dim=0)
        self.keyframe_Rs = torch.cat([self.keyframe_Rs, new_R], dim=0)
        self.keyframe_ts = torch.cat([self.keyframe_ts, new_t], dim=0)
        self.merge()
        self.offload()
        self.local_adjust()

    def get_bundle(
        self, newframe_descs: Tensor, n: int, last_loc: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if last_loc is None:
            _, topk = self.desc_knn(newframe_descs)
            topk = topk.view(-1, 1)
            kf_locs = self.keyframe_locs.type_as(newframe_descs)[topk]
            kf_kpts = self.keyframe_kpts.type_as(newframe_descs)[topk]
            kf_descs = self.keyframe_descs.type_as(newframe_descs)[topk]

            return kf_locs, kf_kpts, kf_descs

        elif len(self) > 5:
            n_closed_used = math.ceil(n // 4)
            nf_idxs, kf_idxs = self.loc_knn(last_loc.view(-1, 3), k=n_closed_used)
            kf_idxs = kf_idxs.view(-1)
            kf_idxs = torch.randperm(len(kf_idxs), device=newframe_descs.device)[:n_closed_used]
            kf_idxs = kf_idxs.view(-1)

            index = torch.arange(len(self), device=newframe_descs.device)
            isin = torch.isin(index, kf_idxs).any(dim=-1)
            index[isin] = len(self)
            recent = torch.sort(index)[0]
            recent = recent[recent < len(self)]
            recent = recent[: n - n_closed_used].view(-1)

            kf_close_locs = self.keyframe_locs.type_as(newframe_descs)[kf_idxs]
            kf_close_kpts = self.keyframe_kpts.type_as(newframe_descs)[kf_idxs]
            kf_close_descs = self.keyframe_descs.type_as(newframe_descs)[kf_idxs]

            kf_temp_locs = self.keyframe_locs.type_as(newframe_descs)[recent]
            kf_temp_kpts = self.keyframe_kpts.type_as(newframe_descs)[recent]
            kf_temp_descs = self.keyframe_descs.type_as(newframe_descs)[recent]

            kf_locs = torch.cat([kf_temp_locs, kf_close_locs])
            kf_kpts = torch.cat([kf_temp_kpts, kf_close_kpts])
            kf_descs = torch.cat([kf_temp_descs, kf_close_descs])

            return kf_locs, kf_kpts, kf_descs

        else:
            kf_locs = self.keyframe_locs.type_as(newframe_descs)[-n:]
            kf_kpts = self.keyframe_kpts.type_as(newframe_descs)[-n:]
            kf_descs = self.keyframe_descs.type_as(newframe_descs)[-n:]

            return kf_locs, kf_kpts, kf_descs


class Track(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        _track: Tensor | None = None
        self.register_buffer("_track", _track)

    @property
    def track(self) -> Tensor:
        if self._track is None:
            return torch.empty(0, 3)
        else:
            return self._track

    @track.setter
    def track(self, locs: Tensor):
        self._track = locs

    def append(self, locs: Tensor):
        locs = locs.detach().cpu()
        self.track = torch.cat([self.track, locs], dim=0)


class LocalGraph(Module):
    def __init__(
        self,
        matcher: Module | Callable,
        msg_queue: Queue | None = None,
        **kwargs,
    ):
        super().__init__()
        self._keyframes = Keyframes(matcher=matcher, msg_queue=msg_queue, **kwargs)
        self._track = Track(matcher=matcher, msg_queue=msg_queue, **kwargs)

        self.msg_queue = msg_queue
        self.matcher = matcher

    @property
    def track(self) -> Tensor:
        return self._track.track

    @property
    def keyframe_locs(self) -> Tensor:
        return self._keyframes.keyframe_locs

    def _get_bundle(
        self, newframe_kpts: Tensor, newframe_descs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor | None]:
        curr_kpts = newframe_kpts
        curr_descs = newframe_descs
        kf_locs: Tensor | None = None
        if len(self._keyframes) > 0:
            n_keyframe_used = min(len(self._keyframes), config.max_n_keyframes_in_local_tracking_queue)

            last_loc = self._track.track[-1].to(newframe_descs.device)

            kf_locs, kf_kpts, kf_descs = self._keyframes.get_bundle(newframe_descs, n_keyframe_used, last_loc)

            n_keyframe_used = kf_kpts.shape[0]
            curr_kpts = torch.cat([kf_kpts, curr_kpts])
            curr_descs = torch.cat([kf_descs, curr_descs])

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

    def _unpack(self, bundle_R: Tensor, bundle_t: Tensor, n_old: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        prev_R = bundle_R[:n_old]
        prev_t = bundle_t[:n_old]
        curr_R = bundle_R[n_old:]
        curr_t = bundle_t[n_old:]
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

    def update(self, newframe_kpts: Tensor, newframe_descs: Tensor):
        bundle_kpts, bundle_descs, bundle_mask, keyframe_locs = self._get_bundle(newframe_kpts, newframe_descs)
        bundle_R, bundle_t = self._bundle_adjust(bundle_kpts, bundle_descs, bundle_mask)
        newframe_R, newframe_t, keyframe_R, keyframe_t = self._unpack(
            bundle_R, bundle_t, keyframe_locs.size(0) if keyframe_locs is not None else 0
        )
        newframe_locs = self._estimate(newframe_R, newframe_t, keyframe_R, keyframe_t, keyframe_locs)

        self._keyframes.update(
            *merger.merge_newframes(newframe_locs, newframe_kpts, newframe_descs, newframe_R, newframe_t)
        )
        self._track.append(newframe_locs)

    def locate(self, kpts: Tensor, descs: Tensor) -> Tensor:
        return NotImplemented
