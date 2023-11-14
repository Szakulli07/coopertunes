"""Module with utilities"""
import logging
import os
import random
from typing import TypeVar

import torch
import torchaudio
import torchaudio.transforms as T
from coloredlogs import ColoredFormatter
from torch import nn

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


def normalize_audio(
    audio: torch.Tensor, from_sample_rate: float, to_sample_rate: float
) -> torch.Tensor:
    # Convert to mono
    if audio.size(0) == 2:
        audio = audio.mean(dim=0, keepdim=True)
    # Resample
    if from_sample_rate != to_sample_rate:
        audio = torchaudio.functional.resample(audio, from_sample_rate, to_sample_rate)
    return audio

def convert_audios2mels(
    audios,
    sample_rate: int = 16_000,
    n_fft: int =1024,
    win_length: int =1024,
    hop_length: int =56,
    n_mels: int =80,
    normalized: bool = False,
):
    # Normalize audio just to be safe
    audios = torch.clamp(audios, min=-1, max=1)

    mels = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        normalized=normalized,
    )(audios)

    return mels  # (b 80 n)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def calc_n_params(module):
    return sum(p.numel() for p in module.parameters())


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
