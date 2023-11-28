from .MelSpecVAE import MelSpecVAE
from .MelGan import MelGanGenerator, MelGanDiscriminator
from .Audio2Mel import Audio2Mel
from .model import Model


def get_model(model_name):
    models_dict = {
        "MelSpecVAE": MelSpecVAE,
        "MelGan": (MelGanGenerator, MelGanDiscriminator),
        "Audio2Mel": Audio2Mel
    }
    return models_dict[model_name]

__all__ = [
    "Model",
    "MelSpecVAE",
    "MelGan",
    "Audio2Mel"
]
