from pathlib import Path
from typing import Any, Optional, Union

from .hparams import HParams


class MelSpecVAEHParams(HParams):

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.lr: float = 2e-4
        self.betas: tuple[float, float] = (0.99, 0.999)
        self.lr_decay: float = 0.999

        self.conv_filters: list[int] = [512, 256, 128, 64, 32]
        self.conv_kernels: list[int] = [3, 3, 3, 3, 3]
        self.conv_strides: list[int | tuple[int]] = [2, 2, 2, 2, 2]
        self.conv_padding: list[int] = [1, 1, 1, 1, 1]
        self.deconv_padding: list[int] = [1, 1, 1, 1, 1]
        self.deconv_out_padding: list[int] = [1, 1, 1, 1, 1]
        self.input_shape: tuple[int, int] = (256, 128)
        self.latent_dim: int = 256
        self.kld_weight: float = 0.00025

        self.hop: int = 256
        self.sample_rate: int = 22_500
        self.n_fft: int = 1024
        self.win_length: int = 1024
        self.hop_length: int = 256
        self.n_mels: int = 256
        self.fmin: float = 0.0
        self.fmax: float = 8_000
        self.normalized: bool = False
        self.segment_len: int = 128

        self.steps_per_ckpt: int = 1_000
        self.batch_size: int = 64
        self.grad_accumulation_steps: int = 1
        self.grad_clip_thresh: float = 0.2

        self.update(hparams)
