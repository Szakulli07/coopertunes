from pathlib import Path
from typing import Any, Optional, Union

from .hparams import HParams


class MelSpecVQVAEHParams(HParams):
    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.lr: float = 2e-4
        self.betas: tuple[float, float] = (0.99, 0.999)
        self.lr_decay: float = 0.999

        self.conv_filters: list[int] = [128, 256, 256]
        self.conv_kernels: list[int] = [3, 3, 3]
        self.conv_strides: list[int | tuple[int]] = [2, 2, 2]
        self.conv_padding: list[int] = [1, 1, 1]
        self.deconv_padding: list[int] = [1, 1, 1]
        self.deconv_out_padding: list[int] = [1, 1, 1]
        self.input_shape: tuple[int, int] = (256, 128)
        self.embedding_dim: int = 256
        self.num_embeddings: int = 128
        self.vq_beta: float = 0.25
        self.vq_weight: float = 1

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

        self.use_fp16: bool = False
        self.use_bf16: bool = False
        self.use_amp: bool = False

        self.update(hparams)

    @property
    def ds_cfg(self):
        return {
            "train_micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": self.grad_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "lr": self.lr,
                "betas": self.betas,
            },
            "scheduler": {
                "type": "ExponentialLR",
                "params": {
                    "gamma": self.lr_decay,
                },
            },
            "fp16": {"enabled": self.use_fp16},
            "bf16": {"enabled": self.use_bf16},
            "amp": {"enabled": self.use_amp},
        }
