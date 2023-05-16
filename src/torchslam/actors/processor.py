from typing import List, Mapping, Tuple

import torch
import torchdata.datapipes as dp
import typer
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
import pandas as pd
from ..ops import detector, matcher, solver, pnp, merger
from .dp import ImageSequenceDataPipe, VideoDataPipe
from websockets.exceptions import ConnectionClosed
from websockets.client import connect  # type: ignore
from websockets.legacy.client import WebSocketClientProtocol
from queue import Queue, Empty
import asyncio
import json
from multiprocessing.managers import Namespace
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from ..utils import log


async def _update_database(websocket: WebSocketClientProtocol, data: Tuple[pd.DataFrame]):
    typer.secho('Updating database', fg=typer.colors.GREEN)
    event = {'type': 'processor-update', 'dataframes': [df.to_json() for df in data]}
    await websocket.send(json.dumps(event))
    await asyncio.sleep(1)


def empty_queue(queue: Queue):
    while True:
        try:
            queue.get(block=True, timeout=1)
        except Empty:
            queue.join()
            break


async def _socket_handler(confs: Mapping, queue: Queue):
    async for websocket in connect(f'ws://{confs["host"]}:{confs["server_port"]}', max_size=2**20 * 128):
        typer.secho('Processor\'s websocket connected', fg=typer.colors.GREEN)
        while True:
            try:
                data = queue.get(block=False)
                await _update_database(websocket, data)
                queue.task_done()
                await asyncio.sleep(1)
            except Empty:
                await asyncio.sleep(1)
            except ConnectionClosed:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(e)
                empty_queue(queue)
                raise e
            except KeyboardInterrupt:
                empty_queue(queue)
                raise KeyboardInterrupt
            except SystemExit:
                empty_queue(queue)
                raise SystemExit


def socket_thread(queue: Queue, confs: Mapping):
    asyncio.run(_socket_handler(confs, queue))


def collate_fn(batch: List[Tensor]) -> Tensor:
    return torch.stack(batch)


def simple_visualize(track: Tensor):
    track = track.detach().cpu()
    assert track.ndim == 2
    assert track.shape[-1] == 3

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title('Track')
    ax.plot(track[:, 0], track[:, 1])
    ax.scatter(track[-1, 0], track[-1, 1], c='r', s=100)
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig('track.png')
    torch.save(track, 'track.pt')
    plt.close(fig)


def run(confs: Mapping, ipc: Namespace | None = None):
    logger.debug('Processor process started')
    queue: Queue[Tuple[pd.DataFrame]] = Queue(maxsize=10)
    if confs['processor_solver_only']:
        logger.debug('Processor solver only')
        processor(None, confs)
        return
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='TorchSLAM-Processor') as executor:
        try:
            logger.debug('Spawning socket thread')
            executor.submit(socket_thread, queue, confs)
            logger.debug('Spawning processor thread')
            executor.submit(processor, queue, confs)
        except KeyboardInterrupt as e:
            executor.shutdown(wait=True, cancel_futures=True)
            raise e
        except SystemExit as e:
            executor.shutdown(wait=True, cancel_futures=True)
            raise e
        except Exception as e:
            logger.error(e)
            executor.shutdown(wait=True, cancel_futures=True)
            raise e
    logger.debug('Processor finished')


def processor(queue: Queue | None, confs: Mapping):
    typer.secho('Processor started', fg=typer.colors.GREEN)
    if confs['input_type'] == 'image_sequence':
        reader: dp.iter.IterDataPipe = ImageSequenceDataPipe(
            confs['input_path'], confs['to_hw'], confs['reverse_sort_filenames']
        )
    elif confs['input_type'] == 'video':
        reader = VideoDataPipe(confs['input_path'], confs['to_hw'])
    else:
        raise ValueError(f'Unknown input type: {confs["input_type"]}')
    logger.debug('Initailizing models')
    n_f = len(reader)
    det_model = detector.SIFT(**confs)
    det_model.to(confs['device'])

    sol = solver.FundamentalMatrix(confs['camera_model'])
    mtc: Module = matcher.RANSACMatcher(sol, solver.sampson_epipolar_distance, **confs)  # type: ignore
    mtc.to(confs['device'])
    logger.debug('Initailizing dataloader')

    frame_iter = reader.sharding_filter().batch(confs['batch_size']).collate(collate_fn=collate_fn)
    rs = MultiProcessingReadingService(num_workers=confs['num_workers'], multiprocessing_context='spawn')
    dl = DataLoader2(frame_iter, reading_service=rs)
    dl.seed(0)
    logger.debug('Starting processor loop')

    track = torch.empty((0, 3), dtype=torch.float32)

    keyframe_locs = torch.empty((0, 3), dtype=torch.float32, device=confs['device'])
    keyframe_kpts = torch.empty((0, confs['num_features'], 3), dtype=torch.float32, device=confs['device'])
    keyframe_descs = torch.empty(
        (0, confs['num_features'], confs['feature_dim']), dtype=torch.float32, device=confs['device']
    )
    keyframe_R = torch.empty((0, 3, 3), dtype=torch.float32, device=confs['device'])
    keyframe_t = torch.empty((0, 3), dtype=torch.float32, device=confs['device'])

    max_n_keyframes = 8
    for f_idx, frame in enumerate(dl):
        if queue is not None:
            typer.secho(
                f'Processing frame {f_idx} / {n_f}; Queue Capacity: {queue.qsize()} / {queue.maxsize}',
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(f'Processing frame {f_idx} / {n_f}', fg=typer.colors.GREEN)
        frame = frame.to(confs['device']).float()

        newframe_kpts, newframe_descs, _ = det_model(frame)
        newframe_kpts = solver.spherical_project_inverse(newframe_kpts)

        curr_kpts = newframe_kpts
        curr_descs = newframe_descs
        n_keyframe_used = 0
        if f_idx > 0 and torch.numel(keyframe_kpts) > 0:
            kf_kpts = keyframe_kpts[-max_n_keyframes:]
            kf_descs = keyframe_descs[-max_n_keyframes:]
            n_keyframe_used = kf_kpts.shape[0]
            curr_kpts = torch.cat([kf_kpts, curr_kpts])
            curr_descs = torch.cat([kf_descs, curr_descs])

        curr_mask = (
            torch.isfinite(curr_descs).all(dim=-1)
            & torch.isfinite(curr_kpts).all(dim=-1)
            & (curr_kpts.norm(dim=-1, p=2) >= 1e-4)
            & (curr_descs.norm(dim=-1, p=2) >= 1e-4)
        )

        curr_R, curr_t = pnp.bundle_adjust(curr_kpts, curr_descs, curr_mask, mtc)

        newframe_locs = torch.empty((0, 3), dtype=torch.float32, device=confs['device'])
        newframe_R = torch.empty((0, 3, 3), dtype=torch.float32, device=confs['device'])
        newframe_t = torch.empty((0, 3), dtype=torch.float32, device=confs['device'])
        if f_idx > 0:
            keyframe_R = curr_R[:n_keyframe_used]
            keyframe_t = curr_t[:n_keyframe_used]
            newframe_R = curr_R[n_keyframe_used:]
            newframe_t = curr_t[n_keyframe_used:]
            kf_locs = keyframe_locs[-n_keyframe_used:]

            log.shapes(
                kf_locs=kf_locs,
                kyframe_R=keyframe_R,
                keyframe_t=keyframe_t,
                newframe_R=newframe_R,
                newframe_t=newframe_t,
            )

            newframe_locs = (
                pnp.creproj(kf_locs.view(-1, 1, 3), keyframe_R, keyframe_t, newframe_R, newframe_t).squeeze(-2).mean(0)
            )

        else:
            newframe_R = curr_R
            newframe_t = curr_t
            R1 = curr_R[:1]
            R2 = curr_R[1:]
            t1 = curr_t[:1]
            t2 = curr_t[1:]
            init_loc = torch.zeros(1, 1, 3, device=curr_R.device, dtype=curr_R.dtype)
            newframe_locs = pnp.reproj(init_loc, R1, t1, R2, t2).squeeze(-2)
            newframe_locs = torch.cat([init_loc.view(-1, 3), newframe_locs], dim=0)

        new_keyframe_kpts, new_keyframe_descs, new_keyframe_R, new_keyframe_t = merger.merge(
            newframe_kpts, newframe_descs, newframe_R, newframe_t
        )  #

        keyframe_locs = torch.cat([keyframe_locs, newframe_locs], dim=0)
        keyframe_kpts = torch.cat([keyframe_kpts, new_keyframe_kpts], dim=0)
        keyframe_descs = torch.cat([keyframe_descs, new_keyframe_descs], dim=0)
        keyframe_R = torch.cat([keyframe_R, new_keyframe_R], dim=0)
        keyframe_t = torch.cat([keyframe_t, new_keyframe_t], dim=0)

        track = torch.cat([track, newframe_locs.cpu()], dim=0)
        simple_visualize(track)
