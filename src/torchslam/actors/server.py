import asyncio
from typing import Any, Iterable, Mapping
from websockets.server import serve

from ..utils import config
from ..ops.database import cozo
from loguru import logger
import pandas as pd
from functools import partial
from multiprocessing.managers import Namespace
from websockets.legacy.server import WebSocketServerProtocol
import typer
import sys
import json


async def update_database(data: Iterable[pd.DataFrame], database: cozo.CozoDB):
    database.update(*data)
    typer.secho('Database updated', fg=typer.colors.GREEN)


async def send_landmarks(websocket, database: cozo.CozoDB):
    raise NotImplementedError
    # data = database.get_landmarks()
    # data = data['xyz'].values.tolist()
    # event = {
    #     'type': 'all-landmark-locations',
    #     'data': {
    #         'xyz': data,
    #     },
    # }
    # await websocket.send(json.dumps(event))
    # await asyncio.sleep(1)


async def handler(websocket: WebSocketServerProtocol, database: cozo.CozoDB):
    typer.secho('Server started listening', fg=typer.colors.GREEN)
    async for event in websocket:
        message = json.loads(event)
        bsize = sys.getsizeof(message) / 2**0
        logger.debug(f'Received message: {bsize:.1f}')
        typer.secho(f'Received message with type {message["type"]}', fg=typer.colors.GREEN)
        if message['type'] == 'processor-update':
            data = [pd.read_json(js) for js in message['dataframes']]
            await update_database(data, database)
        elif message['type'] == 'frontend-request':
            if message['data']['topic'] == 'all-landmark-locations':
                await send_landmarks(websocket, database)


async def server():
    database = cozo.CozoDB()
    async with serve(
        partial(handler, database=database), config['host'], config['server_port'], max_size=2**20 * 128
    ):
        typer.secho('Server started', fg=typer.colors.GREEN)
        await asyncio.Future()


def run(ipc: Namespace | None = None):
    asyncio.run(server())
