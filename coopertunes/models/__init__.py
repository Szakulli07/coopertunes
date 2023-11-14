from .model import Model
from .modelA import ModelA
from .modelB import ModelB
from .MelSpecVAE import MelSpecVAE

def get_model(model_name):
    models_dict = {
        "modelA": ModelA,
        "modelB": ModelB
    }
    return models_dict[model_name]
