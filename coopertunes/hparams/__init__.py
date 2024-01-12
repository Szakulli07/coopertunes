from .hparams import HParams
from .MelSpecVAE import MelSpecVAEHParams
from .MelSpecVQVAE import MelSpecVQVAEHParams


def get_hparams(model_name: str):
    hparams_dict = {
        "MelSpecVAE": MelSpecVAEHParams,
        "MelSpecVQVAE": MelSpecVQVAEHParams,
    }
    return hparams_dict[model_name]


__all__ = [
    "HParams",
    "MelSpecVAEHParams",
    "MelSpecVQVAEHParams",
]
