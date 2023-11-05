from pathlib import Path
from typing import Any, Optional, Union

from .hparams import HParams


class modelBHParams(HParams):
    """Class with hparams for model B"""

    def __init__(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        super().__init__()

        self.exampleA = 200
        self.exampleB = "elpmaxe"

        self.update(hparams)
