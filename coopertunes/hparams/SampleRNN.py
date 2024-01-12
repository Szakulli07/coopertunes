from pathlib import Path
from typing import Any, Optional, Union

from .hparams import HParams


class SampleRNNHParams(HParams):

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.overlap_len: int = 64
        self.q_levels: int = 256
        self.seq_len: int = 1024

        self.audio_len: int = 20_000

        self.update(hparams)
