from pathlib import Path
from typing import Any, Optional, Union

from coopertunes.hparams import HParams


class Audio2MelHParams(HParams):

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.n_fft=1024
        self.hop_length=256
        self.win_length=1024
        self.sampling_rate=22050
        self.n_mel_channels=80
        self.mel_fmin=0.0
        self.mel_fmax=None

        self.update(hparams)
