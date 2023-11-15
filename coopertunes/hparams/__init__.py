from .hparams import HParams
from .MelSpecVAE import MelSpecVAEHParams


def get_hparams(model_name: str):
    hparams_dict = {
        "MelSpecVAE": MelSpecVAEHParams,
    }
    return hparams_dict[model_name]


__all__ = [
    'HParams',
    'MelSpecVAEHParams',
]
