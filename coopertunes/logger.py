'''Module with logger utils class'''
from collections import defaultdict
from statistics import mean
from typing import Any, Literal

import torch
from torch.utils.tensorboard import SummaryWriter

from .hparams import HParams
from .utils import convert_mels2audios_h, log_info, plot_audio, plot_mel


class Logger:
    '''Class for logging training information.
    The logger takes care of providing feedback to `stdout` and to the `tensorboard`
    To add a logger for a model create two functions \
          to log the step and model and to log items to the tensorbord.
    Then register the above functions in `_init_utils_fn`'''

    def __init__(self, model_name: str, hparams: HParams, device: torch.device):
        self.hparams = hparams
        self.model_name = model_name
        self.device = device
        self._logger = SummaryWriter(self.hparams.logs_dir / 'logs')
        self._running_vals = self._reset_running_vals()
        self.log_step, self.log_audio = self._init_utils_fn()

    def log_running_vals_to_tb(self, step):
        for label, vals in self._running_vals.items():
            self._logger.add_scalar(label, mean(vals), step)
        self._running_vals = self._reset_running_vals()

    def update_running_vals(self, vals: dict[str, Any], prefix: str | None = None):
        for key, val in vals.items():
            if prefix is not None:
                key = f'{prefix}/{key}'
            self._running_vals[key].append(val)

    def _reset_running_vals(self):
        return defaultdict(list)

    def _init_utils_fn(self):
        log_fn_dict = {
            'melspecvae': (self._log_step_vae, self._log_mel_batch),
            'melspecvqvae': (self._log_step_vqvae, self._log_mel_batch),
        }
        return log_fn_dict[self.model_name]

    def _log_step_vae(
        self,
        epoch: int,
        step: int,
        prefix: Literal['training', 'validation'] = 'training',
    ):
        log_info(
            'Epoch: %d | Step: %d | LossRecon: %.4f | LossKLD: %.4f | StepTime: %.2f[s]',
            epoch,
            step,
            mean(self._running_vals[f'{prefix}/recon']),
            mean(self._running_vals[f'{prefix}/kld']),
            mean(self._running_vals[f'{prefix}/step_time']),
        )

    def _log_step_vqvae(
        self,
        epoch: int,
        step: int,
        prefix: Literal['training', 'validation'] = 'training',
    ):
        log_info(
            'Epoch: %d | Step: %d | LossRecon: %.4f | LossVQ: %.4f | StepTime: %.2f[s]',
            epoch,
            step,
            mean(self._running_vals[f'{prefix}/recon']),
            mean(self._running_vals[f'{prefix}/vq']),
            mean(self._running_vals[f'{prefix}/step_time']),
        )

    def _log_mel(
        self,
        mel: torch.Tensor,
        audio_type: Literal['target', 'output', 'generated'],
        index: int,
        step: int,
    ):
        audio_name = f'{audio_type}_{index}'

        self._logger.add_image(
            f'mel/{audio_name}', plot_mel(mel.cpu()), step, dataformats='HWC'
        )

        audio = convert_mels2audios_h(mel.cpu(), self.hparams)

        self._logger.add_image(
            f'wav/{audio_name}', plot_audio(audio), step, dataformats='HWC'
        )

        self._logger.add_audio(
            f'audio/{audio_name}', audio, step, sample_rate=self.hparams.sample_rate
        )

    def _log_mel_batch(
        self, batch, step: int, audio_type: Literal['target', 'output']
    ):
        batch = [audio.to(self.device) for audio in batch]
        for audio_index, audio in enumerate(batch):
            self._log_mel(audio, audio_type, audio_index, step)
