from .model import Model
from .MelSpecVAE import MelSpecVAE

def get_model(model_name):
    models_dict = {
        "MelSpecVAE": MelSpecVAE,
    }
    return models_dict[model_name]
