from coopertunes.models.Audio2Mel import Audio2Mel
from coopertunes.models.MelGan import MelGanDiscriminator, MelGanGenerator
from coopertunes.models.MelSpecVAE import MelSpecVAE
from coopertunes.models.MelSpecVQVAE import MelSpecVQVAE
from coopertunes.models.model import Model
from coopertunes.models.PerformanceRNN import PerformanceRNN


def get_model(model_name):
    models_dict = {
        "MelSpecVAE": MelSpecVAE,
        "MelSpecVQVAE": MelSpecVQVAE,
        "MelGanGenerator": MelGanGenerator,
        "MelGanDiscriminator": MelGanDiscriminator,
        "Audio2Mel": Audio2Mel,
        "PerformanceRNN": PerformanceRNN,
    }
    return models_dict[model_name]


__all__ = [
    "Model",
    "MelSpecVAE",
    "MelSpecVQVAE",
    "MelGanGenerator",
    "MelGanDiscriminator",
    "Audio2Mel",
    "PerformanceRNN",
]
