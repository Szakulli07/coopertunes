from pathlib import Path
from typing import Any, Optional, Union
from .hparams import HParams


class MelSpecVAEHParams(HParams):

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.conv_filters: list[int] = [512, 256, 128, 64, 32]
        self.conv_kernels: list[int] = [3, 3, 3, 3, 3]
        self.conv_strides: list[int | tuple[int]] = [2, 2, 2, 2, 2]
        self.conv_padding: list[int]  = [1, 1, 1, 1, 1]
        self.deconv_padding: list[int]  = [1, 1, 1, 1, 1]
        self.deconv_out_padding: list[int]  = [1, 1, 1, 1, 1]
        self.input_shape: tuple[int] = (256, 128)
        self.latent_dim: int = 64
        self.recon_loss_weight: int = 1000000

        self.hop = 256
        self.sample_rate = 44100
        self.n_fft = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.n_mels = 256
        self.normalized = False
        self.segment_len = 128

        self.update(hparams)
