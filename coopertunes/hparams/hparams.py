import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from ..utils import log_error


class HParams(ABC):
    """Base class with hyperparameters"""

    @abstractmethod
    def __init__(self) -> None:
        self.train_data_dirs: list[Path] = [Path("data/train")]
        self.valid_data_dirs: list[Path] = [Path("data/valid")]
        self.checkpoints_dir: Path = Path("models/ckpt")
        self.logs_dir: Path = Path("models/log")

        self.base_checkpoint: int | None = None
        self.steps_per_log: int = 100
        self.steps_per_ckpt: int = 5_000
        self.loader_num_workers: int = 4
        self.batch_size: int = 1
        self.valid_batch_size: int = 1

    def update(self, hparams: Optional[Union[Path, dict[str, Any]]] = None):
        if hparams is not None:
            if isinstance(hparams, dict):
                self._update_with_dict(hparams)
            else:
                self._update_with_file(hparams)
            self.train_data_dirs = [Path(path) for path in self.train_data_dirs]
            self.valid_data_dirs = [Path(path) for path in self.valid_data_dirs]
            self.checkpoints_dir = Path(self.checkpoints_dir)
            self.logs_dir = Path(self.logs_dir)

    def _update_with_file(self, hparams_fp: Path):
        suffix = hparams_fp.suffix
        assert suffix == ".json", log_error("Hparams file must be json file")
        with open(hparams_fp, "r", encoding="utf-8") as fh:
            hparams_dict = json.load(fh)
        self._update_with_dict(hparams_dict)

    def _update_with_dict(self, hparams_dict: dict[str, Any]):
        for key, value in hparams_dict.items():
            if key not in self.__dict__:
                raise ValueError(f"hparam {key} does not exist")
            self.__dict__[key] = value

    def dumps_to_file(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = self.checkpoints_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "hparams.json", "w", encoding="utf-8") as f:
            f.write(self._dumps())

    def _dumps(self):
        data = {k: self._repr_object(v) for k, v in self.__dict__.items()}
        return json.dumps(data, indent=4, sort_keys=True)

    def _repr_object(self, v):
        if isinstance(v, list):
            return [self._repr_object(v_) for v_ in v]
        if isinstance(v, Path):
            return str(v)
        return v

    def __repr__(self) -> str:
        return self._dumps()
