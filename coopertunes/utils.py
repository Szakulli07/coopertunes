"""Module with utilities"""
import logging
import os
import random
from typing import TypeVar

import librosa
import matplotlib.pyplot as plt
import scipy.io.wavfile
import numpy as np
import torch
from coloredlogs import ColoredFormatter
from torch import nn

from coopertunes.distributed import global_rank, local_rank

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

_SAMPLE_NORMALIZATION_FACTOR = 32768


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


def normalize_audio(audio, from_sample_rate: float, to_sample_rate: float):
    # Convert to mono
    if audio.ndim == 2:
        audio = librosa.to_mono(audio)
    # Resample
    if from_sample_rate != to_sample_rate:
        audio = librosa.resample(
            audio, orig_sr=from_sample_rate, target_sr=to_sample_rate
        )
    return audio


def convert_audios2mels(
    audios,
    sample_rate,
    n_mels=80,
    hop_len=256,
    n_fft=1024,
    win_len=1024,
    fmin=0.0,
    fmax=8000.0,
):
    spectrograms = np.abs(
        librosa.stft(y=audios, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    )
    mels = torch.FloatTensor(
        librosa.feature.melspectrogram(
            S=spectrograms,
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
    )

    mels = torch.clamp(mels, 1e-5, None)
    mels = torch.log(mels)
    mels = (mels + 5.0) / 5.0
    return mels  # b c t


def convert_audios2mels_h(audios, hparams):
    return convert_audios2mels(
        audios,
        hparams.sample_rate,
        hparams.n_mels,
        hparams.hop_length,
        hparams.n_fft,
        hparams.win_length,
        hparams.fmin,
        hparams.fmax,
    )


def get_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def convert_mels2audios(
    mels,
    sample_rate,
    n_griffin_lim_iter=16,
    hop_len=256,
    n_fft=1024,
    win_len=1024,
    fmin=0.0,
    fmax=8000.0,
):
    # Clip exotic values (they are sometimes produced by a model)
    mels = torch.clamp(mels, -5.0, 5.0)

    mels = mels * 5.0 - 5.0
    mels = np.exp(mels)

    spectrograms = librosa.feature.inverse.mel_to_stft(
        M=mels.numpy(), power=1, sr=sample_rate, n_fft=n_fft, fmin=fmin, fmax=fmax
    )
    audios = torch.FloatTensor(
        librosa.griffinlim(
            S=spectrograms,
            n_iter=n_griffin_lim_iter,
            hop_length=hop_len,
            win_length=win_len,
        )
    )

    audios = torch.clamp(audios, -1, 1)

    return audios  # b t


def convert_mels2audios_h(mels, hparams):
    return convert_mels2audios(
        mels,
        hparams.sample_rate,
        16,
        hparams.hop_length,
        hparams.n_fft,
        hparams.win_length,
        hparams.fmin,
        hparams.fmax,
    )


def _fig2numpy(fig):
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_mel(mel, out_fp=None):
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(mel, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("frame")
    plt.ylabel("channel")
    plt.tight_layout()

    fig.canvas.draw()
    data = _fig2numpy(fig)
    if out_fp:
        plt.savefig(out_fp)
    plt.close()
    return data


def plot_audio(audio, out_fp=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(audio, linewidth=0.1, alpha=0.7)
    plt.ylim(-1, 1)
    plt.xlabel("sample")
    plt.ylabel("amplitude")
    plt.tight_layout()

    fig.canvas.draw()
    data = _fig2numpy(fig)
    if out_fp:
        plt.savefig(out_fp)
    plt.close()
    return data


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def calc_n_params(module):
    return sum(p.numel() for p in module.parameters())


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * _SAMPLE_NORMALIZATION_FACTOR).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


def dconv_same_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class PixelNormalization(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * (((x**2).mean(dim=1, keepdim=True) + self.eps).rsqrt())
