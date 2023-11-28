from ..models import Audio2Mel
from ..hparams import Audio2MelHParams

import torch

class Audio2MelSupervisor:
    def __init__(self, model: Audio2Mel, device: torch.device, hparams: Audio2MelHParams):
        self.model = Audio2Mel(hparams).to(device)
        self.device = device
        

    def convert(self, audio):
        """
        Performs audio to mel conversion (See Audio2Mel in mel2wav/modules.py)
        Args:
            audio (torch.tensor): PyTorch tensor containing audio (batch_size, timesteps)
        Returns:
            torch.tensor: log-mel-spectrogram computed on input audio (batch_size, 80, timesteps)
        """
        return self.model(audio.unsqueeze(1).to(self.device))
