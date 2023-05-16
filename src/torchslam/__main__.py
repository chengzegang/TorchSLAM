from typing import Literal
import typer
import torchslam.__about__
from . import actors
from .utils.config import Configuration
import os
import torch
from loguru import logger
import sys
import multiprocessing as mp
import signal
import toml

app = typer.Typer(pretty_exceptions_enable=False)


class Signal(Exception):
    pass


def sigint_handler(signum, frame):
    raise Signal


signal.signal(signal.SIGINT, sigint_handler)


@app.command()
def version():
    """Show version"""
    typer.echo(f"torchslam version: {torchslam.__about__.__version__}")


@app.command()
def serve(host: str = 'localhost', port: int = 7000):
    """Run server only"""
    typer.echo("Running server only\n")
    conf = Configuration(host=host, server_port=port, actors=('server',))
    actors.spawn(conf)


@app.command()
def solve(input: str):
    """Run solver only"""
    typer.echo("Running solver only\n")
    input_type: Literal['video', 'image_sequence'] = 'video'
    if os.path.isdir(input):
        input_type = 'image_sequence'
    elif os.path.isfile(input):
        input_type = 'video'
    device: str = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    conf = Configuration(input, input_type, device=device, actors=('processor',), processor_solver_only=True)
    actors.spawn(conf)


@app.command()
def wtf(input: str):
    """Quick start"""
    typer.secho("WTF is this configuration? JUST RUN IT!\n", fg=typer.colors.RED, bold=True)
    input_type: Literal['video', 'image_sequence'] = 'video'
    if os.path.isdir(input):
        input_type = 'image_sequence'
    elif os.path.isfile(input):
        input_type = 'video'
    device: str = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    conf = Configuration(input, input_type, device=device)
    actors.spawn(conf)


@app.command()
def run(config_filepath: str):
    """Run SLAM"""
    typer.echo("Running SLAM\n")
    conf = Configuration(**toml.load(config_filepath))
    actors.spawn(conf)


if __name__ == "__main__":
    typer.secho(
        "========================================================================",
        fg=typer.colors.BRIGHT_YELLOW,
        bold=True,
    )
    typer.secho(
        ">>> Welcome to the TorchSLAM, a distributed SLAM System built on Pytorch",
        fg=typer.colors.BRIGHT_YELLOW,
        bold=True,
    )
    typer.secho(
        "========================================================================\n",
        fg=typer.colors.BRIGHT_YELLOW,
        bold=True,
    )
    app()
