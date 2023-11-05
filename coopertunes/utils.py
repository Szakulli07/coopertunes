"""Module with utilities"""
import logging
import os
import random
from typing import TypeVar

import torch
from coloredlogs import ColoredFormatter

from .distributed import global_rank, local_rank

L = TypeVar("L")


AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3"]


_LOGGER = logging.getLogger(__package__)
_LOGGER.propagate = False
_LOGGER.setLevel(logging.INFO)

_FORMATTER = ColoredFormatter(
    fmt=f"%(asctime)s :: %(levelname)s :: GR={global_rank()};LR={local_rank()} :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_CONSOLE_HANDLER = logging.StreamHandler()
_CONSOLE_HANDLER.setFormatter(_FORMATTER)
_LOGGER.addHandler(_CONSOLE_HANDLER)


def log_debug(*args, **kwargs):
    _LOGGER.debug(*args, **kwargs)


def log_info(*args, **kwargs):
    _LOGGER.info(*args, **kwargs)


def log_warning(*args, **kwargs):
    _LOGGER.warning(*args, **kwargs)


def log_error(*args, **kwargs):
    _LOGGER.error(*args, **kwargs)


def setup_cuda_debug(cuda_debug_mode: bool = False):
    if cuda_debug_mode:
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"
        os.environ["NCCL_P2P_LEVEL"] = "NVL"
        os.environ["NCCL_P2P_DISABLE"] = "1"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def calc_n_params(module):
    return sum(p.numel() for p in module.parameters())
