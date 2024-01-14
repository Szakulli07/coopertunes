import os 
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union
from datatools.miditools import EventSeq, ControlSeq

from .hparams import HParams

class PerformanceRNNHParams(HParams):

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        #Model
        self.init_dim: int = 32
        self.event_dim: int = EventSeq.dim()
        self.control_dim: int = ControlSeq.dim()
        self.hidden_dim: int = 512
        self.gru_layers: int = 3
        self.gru_dropout: float = 0.3

        #Training
        self.default_checkpoint: Path = os.path.join("coopertunes", "checkpoints", "PerformanceRNN", "default_checkpoint.pt")
        self.learning_rate: float = 0.001
        self.batch_size: int = 64
        self.window_size: int = 200
        self.stride_size: int = 10
        self.use_transposition: bool = False
        self.control_ratio: float = 1.0
        self.teacher_forcing_ratio: float = 1.0
        

