from .hparams import HParams
from .MelSpecVAE import MelSpecVAEHParams
from .MelGan import MelGanHParams
from .Audio2Mel import Audio2MelHParams


def get_hparams(model_name: str):
    hparams_dict = {
        "MelSpecVAE": MelSpecVAEHParams,
        "MelGan": MelGanHParams,
        "Audio2Mel": Audio2MelHParams,
    }
    return hparams_dict[model_name]


__all__ = [
    "HParams",
    "MelSpecVAEHParams",
    "Audio2MelHParams",
    "MelGanHParams"
]
