from coopertunes.supervisors.Audio2Mel import Audio2MelSupervisor
from coopertunes.supervisors.GANSynth import GANSynthSupervisor
from coopertunes.supervisors.MelGan import MelGanSupervisor
from coopertunes.supervisors.MelSpecVAE import MelSpecVAESupervisor
from coopertunes.supervisors.MelSpecVQVAE import MelSpecVQVAESupervisor

__all__ = [
    "MelSpecVAESupervisor",
    "MelSpecVQVAESupervisor",
    "MelGanSupervisor",
    "Audio2MelSupervisor",
    "GANSynthSupervisor",
]
