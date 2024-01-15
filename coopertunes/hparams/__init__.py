from coopertunes.hparams.Audio2Mel import Audio2MelHParams
from coopertunes.hparams.GANSynth import GANSynthHParams
from coopertunes.hparams.hparams import HParams
from coopertunes.hparams.MelGan import MelGanHParams
from coopertunes.hparams.MelSpecVAE import MelSpecVAEHParams
from coopertunes.hparams.MelSpecVQVAE import MelSpecVQVAEHParams
from coopertunes.hparams.PerformanceRNN import PerformanceRNNHParams


def get_hparams(model_name: str):
    hparams_dict = {
        "MelSpecVAE": MelSpecVAEHParams,
        "MelSpecVQVAE": MelSpecVQVAEHParams,
        "MelGan": MelGanHParams,
        "Audio2Mel": Audio2MelHParams,
        "PerformanceRNN": PerformanceRNNHParams,
        "GANSynthHParams": GANSynthHParams,
    }
    return hparams_dict[model_name]


__all__ = [
    "HParams",
    "MelSpecVAEHParams",
    "MelSpecVQVAEHParams",
    "Audio2MelHParams",
    "MelGanHParams",
    "PerformanceRNNHParams",
    "GANSynthHParams",
]
