from typing import Callable, Mapping
from . import server, processor, optimizer
import multiprocessing as mp
from loguru import logger
from concurrent.futures import ProcessPoolExecutor


def spawn(confs: Mapping):
    mp.set_start_method('spawn')
    acts = []
    for a in confs['actors']:
        if a == 'server':
            acts.append(server.run)
        if a == 'processor':
            acts.append(processor.run)
        if a == 'optimizer':
            acts.append(optimizer.run)
    if len(acts) == 1:
        acts[0](confs)
    with mp.Manager() as manager, ProcessPoolExecutor(max_workers=2) as executor:
        try:
            ipc = manager.Namespace()

            for fn in acts:
                executor.submit(fn, confs, ipc)

        except KeyboardInterrupt:
            executor.shutdown(wait=True, cancel_futures=True)
            raise KeyboardInterrupt
        except SystemExit:
            executor.shutdown(wait=True, cancel_futures=True)
            raise SystemExit
        except Exception as e:
            logger.error(e)
            executor.shutdown(wait=True, cancel_futures=True)
            raise e
