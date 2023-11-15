from .MelSpecVAE import MelSpecVAE
from .model import Model


def get_model(model_name):
    models_dict = {
        "MelSpecVAE": MelSpecVAE,
    }
    return models_dict[model_name]

__all__ = [
    'Model',
    'MelSpecVAE',
]
