from .MelSpecVAE import MelSpecVAE
from .MelSpecVQVAE import MelSpecVQVAE
from .model import Model


def get_model(model_name):
    models_dict = {
        "MelSpecVAE": MelSpecVAE,
        "MelSpecVQVAE": MelSpecVQVAE,
    }
    return models_dict[model_name]


__all__ = [
    "Model",
    "MelSpecVAE",
    "MelSpecVQVAE"
]
