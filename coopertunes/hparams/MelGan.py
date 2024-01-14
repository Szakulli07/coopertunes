import os
from pathlib import Path
from typing import Any, Optional, Union

from .hparams import HParams


class MelGanHParams(HParams):
    """
    Parameters for MelGan and MelGanSupervisor.
    """

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.summary_path = os.path.join("models", "summary")
        self.default_checkpoint = os.path.join("coopertunes", "checkpoints", "MelGan", "multi_speaker.pt")
        Path(
            "coopertunes/checkpoints/MelGan/multi_speaker.pt")

        # Model
        self.seq_len: int = 8192
        self.sampling_rate: int = 22050
        self.segment_length_ratio: int = 4
        self.ngf: int = 32
        self.n_residual_layers: int = 3
        self.ndf: int = 16
        self.n_layers_D: int = 4
        self.cond_disc: bool = False
        self.num_D: int = 3
        self.n_mel_channels: int = 80
        self.batch_size: int = 64
        self.downsamp_factor: int = 4
        self.epochs: int = 3000
        self.lambda_feat: int = 10
        self.log_interval: int = 100
        self.n_test_samples: int = 8
        self.save_interval: int = 1000
        self.generator_ratios: list = [8, 8, 2, 2]
        self.learning_rate = 1e-4
        self.adam_betas = (0.5, 0.9)

        self.update(hparams)
