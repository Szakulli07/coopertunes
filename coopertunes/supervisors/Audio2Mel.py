import torch

from coopertunes.models import Audio2Mel
from coopertunes.hparams import Audio2MelHParams


class Audio2MelSupervisor:
    def __init__(self, model: Audio2Mel, device: torch.device, hparams: Audio2MelHParams):
        self.model = model.to(device)
        self.hparams = hparams
        self.device = device

    def convert(self, audio):
        """
        Args:
            audio (torch.tensor): PyTorch tensor containing audio (batch_size, timesteps)
        Returns:
            torch.tensor: log-mel-spectrogram
                computed on input audio (batch_size, mel bank filters, timesteps)
        """
        return self.model(audio.unsqueeze(1).to(self.device))
