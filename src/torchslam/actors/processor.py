from typing import List, Mapping, Tuple

import torch
import torchdata.datapipes as dp
import typer
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
import pandas as pd

from torchslam.ops.functional.proj import spherical
from ..ops import detector, graph, matcher, solver
from ..ops.dp import ImageSequenceDataPipe, VideoDataPipe
from websockets.exceptions import ConnectionClosed
from websockets.client import connect  # type: ignore
from websockets.legacy.client import WebSocketClientProtocol
from queue import Queue, Empty
import asyncio
import json
from multiprocessing.managers import Namespace
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt


def build_socket_event(
    keyframe_locs: Tensor, keyframe_kpts: Tensor, keyframe_descs: Tensor, keyframe_R: Tensor, keyframe_t: Tensor
) -> dict:
    event = {
        'type': 'processor-update-keyframes',
        'data': {
            'keyframe_locs': keyframe_locs.tolist(),
            'keyframe_kpts': keyframe_kpts.tolist(),
            'keyframe_descs': keyframe_descs.tolist(),
            'keyframe_R': keyframe_R.tolist(),
            'keyframe_t': keyframe_t.tolist(),
        },
    }
    return event


async def _update_database(websocket: WebSocketClientProtocol, data: Tuple[Tensor]):
    typer.secho('Updating database', fg=typer.colors.GREEN)
    event = build_socket_event(*data)  # type: ignore
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


def simple_visualize(track: Tensor, kf_locs: Tensor):
    track = track.detach().cpu()
    kf_locs = kf_locs.detach().cpu()
    assert track.ndim == 2
    assert track.shape[-1] == 3

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title('Track')
    ax.plot(track[:, 0], track[:, 1], alpha=0.5, c='gray')
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.scatter(track[-1, 0], track[-1, 1], c='r', s=300)
    ax.scatter(kf_locs[:, 0], kf_locs[:, 1], c='g', s=100)
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

    sol = solver.Fundamental(confs['camera_model'])
    mtc: Module = matcher.RANSACMatcher(sol, solver.sampson, **confs)  # type: ignore
    mtc.to(confs['device'])
    logger.debug('Initailizing dataloader')

    frame_iter = reader.sharding_filter().batch(confs['batch_size']).collate(collate_fn=collate_fn)
    rs = MultiProcessingReadingService(num_workers=confs['num_workers'], multiprocessing_context='spawn')
    dl = DataLoader2(frame_iter, reading_service=rs)
    dl.seed(0)
    logger.debug('Starting processor loop')

    lg = graph.LocalGraph(**confs, matcher=mtc)
    lg.to(confs['device'])

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
        newframe_kpts = spherical(newframe_kpts, True)
        lg.update(newframe_kpts, newframe_descs)
        simple_visualize(lg.track, lg.keyframe_locs)
