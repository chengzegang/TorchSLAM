from typing import Mapping, Tuple

import typer
from loguru import logger
import pandas as pd
from websockets.exceptions import ConnectionClosed
from websockets.client import connect  # type: ignore
from websockets.legacy.client import WebSocketClientProtocol
from queue import Queue, Empty
import asyncio
import json


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
