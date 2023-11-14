from .hparams import HParams
from .hparamsA import modelAHParams
from .hparamsB import modelBHParams
from .MelSpecVAE import MelSpecVAEHParams


def get_hparams(model_name: str):
    hparams_dict = {
        "modelA": modelAHParams,
        "modelB": modelBHParams
    }
    return hparams_dict[model_name]
