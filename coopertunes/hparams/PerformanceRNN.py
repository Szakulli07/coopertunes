import os 
import numpy as np
from pathlib import Path
from typing import Any, Optional, Union
from coopertunes.datatools.miditools import EventSeq, ControlSeq

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
        self.train_data_dirs:list[Path] = [os.path.join("data", "processed", "midi", "train_data")]
        self.default_checkpoint: Path = os.path.join("coopertunes", "checkpoints", "performancernn_pretrained.pt")
        self.learning_rate: float = 0.001
        self.batch_size: int = 64
        self.window_size: int = 200
        self.stride_size: int = 10
        self.use_transposition: bool = False
        self.control_ratio: float = 1.0
        self.teacher_forcing_ratio: float = 1.0
        self.reset_optimizer: bool = False
        self.enable_logging: bool = True

        #Generation
        self.max_len: int = 1000 
        self.greedy_ratio: float = 1.0
        self.stochastic_beam_search: bool = False
        self.beam_size: int = 0
        self.temperature: float = 1.0
