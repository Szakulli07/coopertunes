from pathlib import Path
from typing import Any, Optional, Union

from .hparams import HParams


class MelGanHParams(HParams):

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.seq_len= 8192
        self.sampling_rate = 22050
        self.ngf = 32
        self.n_residual_layers = 3
        self.ndf= 16
        self.n_layers_D= 4
        self.cond_disc= False
        self.num_D= 3
        self.n_mel_channels= 80
        self.batch_size= 64
        self.downsamp_factor= 4
        self.epochs= 3000
        self.lambda_feat= 10
        self.log_interval= 100
        self.n_test_samples= 8
        self.save_interval= 1000
        self.load_path= ""
        self.data_path= ""
        self.save_path= "summary"

        self.update(hparams)
