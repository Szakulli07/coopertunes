import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn

from coopertunes.hparams import Audio2MelHParams


class Audio2Mel(nn.Module):
    def __init__(self, hparams: Audio2MelHParams):
        super().__init__()
        window = torch.hann_window(hparams.win_length).float()
        mel_basis = librosa_mel_fn(
            sr=hparams.sampling_rate,
            n_fft=hparams.n_fft,
            n_mels=hparams.n_mel_channels,
            fmin=hparams.mel_fmin,
            fmax=hparams.mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = hparams.n_fft
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.n_mel_channels = hparams.n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec

    def inference(self, audio):
        self.forward(audio)
