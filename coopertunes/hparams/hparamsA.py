from pathlib import Path
from typing import Any, Optional, Union
from .hparams import HParams


class modelAHParams(HParams):
    """Class with hparams for model A"""

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.exampleA = 100
        self.exampleB = "example"

        self.update(hparams)
