from typing import List, Mapping, Tuple

import torch
import torchdata.datapipes as dp
import typer
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
import pandas as pd
from ..ops import detector, matcher, solver, pnp
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
    prev_locs = torch.as_tensor([0, 0, 0], dtype=torch.float32, device=confs['device']).view(1, 3)
    prev_kpts = torch.zeros((1, 0, 2), dtype=torch.float32, device=confs['device'])
    prev_descs = torch.zeros((1, 0, 512), dtype=torch.float32, device=confs['device'])
    track = []
    for f_idx, frame in enumerate(dl):
        if queue is not None:
            typer.secho(
                f'Processing frame {f_idx} / {n_f}; Queue Capacity: {queue.qsize()} / {queue.maxsize}',
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(f'Processing frame {f_idx} / {n_f}', fg=typer.colors.GREEN)
        frame = frame.to(confs['device']).float()

        kpts, descs, _ = det_model(frame)
        kpts = solver.spherical_project_inverse(kpts)

        curr_kpts, curr_descs = kpts, descs
        if f_idx > 0:
            curr_kpts = torch.cat([prev_kpts, curr_kpts], dim=0)
            curr_descs = torch.cat([prev_descs, curr_descs], dim=0)
        mask = (
            torch.isfinite(curr_descs).all(dim=-1)
            & torch.isfinite(curr_kpts).all(dim=-1)
            & (curr_kpts.norm(dim=-1, p=2) >= 1e-4)
            & (curr_descs.norm(dim=-1, p=2) >= 1e-4)
        )
        logger.debug(f'shapes of kpts: {curr_kpts.shape}, descs: {curr_descs.shape}, mask: {mask.shape}')
        R, t = pnp.bundle_adjust(curr_kpts, curr_descs, mask, mtc)

        curr_locs = torch.empty((0, 3), dtype=torch.float32, device=confs['device'])
        if f_idx > 0:
            n_p = len(prev_locs)
            R1 = R[:n_p]
            R2 = R[n_p:]
            t1 = t[:n_p]
            t2 = t[n_p:]
            logger.debug(f'shapes of R1: {R1.shape}, R2: {R2.shape}, t1: {t1.shape}, t2: {t2.shape}')
            prev_locs = prev_locs.view(-1, 1, 3)
            curr_locs = pnp.reproj(prev_locs, R1, t1, R2, t2).squeeze(1)
        else:
            prev_locs = prev_locs.view(-1, 1, 3)
            curr_locs = pnp.proj(prev_locs, R, t).squeeze(1)

        logger.debug(f'shape of curr_locs: {curr_locs.shape}')
        prev_locs = curr_locs.clone()
        prev_kpts = kpts.clone()
        prev_descs = descs.clone()

        # merge_locs = curr_locs.nanmean(dim=0, keepdim=True)
        track.append(curr_locs.detach().cpu())
        track = [torch.cat(track, dim=0)]
        logger.debug(f'track shape: {track[0].shape}')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Track')
        ax.plot(track[0][..., 0], track[0][..., 1], track[0][..., 2])
        ax.scatter(track[0][-1, 0], track[0][-1, 1], track[0][-1, 2], c='r', s=100)
        ax.set_aspect('equal')
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.savefig('track.png')
        torch.save(track, 'track.pt')
        plt.close(fig)
