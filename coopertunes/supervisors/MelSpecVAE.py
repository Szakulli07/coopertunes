import os
import time
from statistics import mean

import deepspeed
import torch
from einops import rearrange
from torch.utils.data import DataLoader

from coopertunes.datasets import MelDataset
from coopertunes.distributed import (
    get_world_size,
    global_rank,
    is_global_leader
)
from coopertunes.hparams import MelSpecVAEHParams
from coopertunes.logger import Logger
from coopertunes.models import MelSpecVAE
from coopertunes.utils import log_info


class MelSpecVAESupervisor:
    """Supervisor for MelSpecVAESupervisor
    After init you can launch training with `train` method"""

    def __init__(self, model: MelSpecVAE, device: torch.device, hparams: MelSpecVAEHParams):
        self.model = model
        self.device = device
        self.hparams = hparams

        self.epoch = 1
        self.step = 1

        if is_global_leader():
            self._logger = Logger("melspecvae", self.hparams, device)

        self.train_dl, self.val_dl = self._build_loaders()
        self.engines = self._build_engines()

        if self.hparams.base_checkpoint:
            self._load_checkpoint()
        else:
            log_info("Initilizing fresh training")

        if is_global_leader():
            self.hparams.dumps_to_file()

    def train(self):

        if is_global_leader():
            log_info('Started training')

        for engine in self.engines.values():
            engine.train()

        train_dl_iter = self._make_infinite_epochs(self.train_dl)

        # Training loop
        while True:
            torch.cuda.synchronize()
            # break training if reached total_steps
            if self.step >= self.hparams.total_steps:
                log_info("Max steps reached. Training finished")
                break

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()

            start = time.time()

            self.engines['model'].zero_grad()
            for _ in range(self.hparams.grad_accumulation_steps):
                batch = next(train_dl_iter)
                mels = batch["mels"].to(self.device)

                reconstruct, x, mu, log_var = self.engines['model'](mels)
                loss = self.engines['model'].loss_function(reconstruct, x, mu, log_var)
                self.engines['model'].backward(loss['loss'])

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.engines['model'].parameters(), self.hparams.grad_clip_thresh
            )

            self.engines['model'].step()

            torch.cuda.synchronize()

            stats = {
                'loss': loss["loss"].item(),
                'recon': loss["recon"].item(),
                'kld': loss["kld"].item(),
                'grad_norm': grad_norm.item(),
                'learning_rate': self.engines['model'].get_lr()[0],
                'step_time': (time.time() - start),
            }
            if is_global_leader():
                self._log_train_stats(stats)

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()
                if is_global_leader():
                    for engine in self.engines.values():
                        engine.eval()
                    self.eval()
                    for engine in self.engines.values():
                        engine.train()

            self.step += 1

    @torch.inference_mode()
    def eval(self):
        log_info('Started validation')

        start = time.time()
        loss_list = []
        recon_list = []
        kld_list = []

        for i, batch in enumerate(self.val_dl):
            mels = batch["mels"].to(self.device)
            reconstruct, x, mu, log_var = self.engines['model'](mels)
            loss = self.engines['model'].loss_function(reconstruct, x, mu, log_var)

            loss_list.append(loss["loss"].item())
            recon_list.append(loss["recon"].item())
            kld_list.append(loss["kld"].item())

            if i == 0:
                reconstructs = [None] * batch["mels"].shape[0]
                for i, recon in enumerate(rearrange(reconstruct, 'n 1 c t ->n c t')):
                    reconstructs[i] = recon.detach()

        stats = {
            'loss': mean(loss_list),
            'recon': mean(recon_list),
            'kld': mean(kld_list),
            'step_time': (time.time() - start),
        }

        self._logger.log_audio(
            batch=reconstructs, step=self.step, audio_type='output')
        self._logger.update_running_vals(stats, 'validation')
        self._logger.log_step(self.epoch, self.step, prefix='validation')
        self._logger.log_running_vals_to_tb(self.step)
        log_info('Validation ends')

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_dataset = self._build_dataset(training=True)
        train_dl = self._create_dataloader(train_dataset, training=True)

        if not is_global_leader():
            val_dl = None
        else:
            val_dataset = self._build_dataset(training=False)
            val_dl = self._create_dataloader(val_dataset, training=False)

            for i, batch in enumerate(val_dl):
                if i == 0:
                    mels = [None] * batch["mels"].shape[0]
                    for j, mel in enumerate(rearrange(batch["mels"], 'n 1 c t ->n c t')):
                        mels[j] = mel
                    self._logger.log_audio(mels, step=0, audio_type='target')

        return train_dl, val_dl

    def _build_engines(self):
        engine, *_ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=self.hparams.ds_cfg,
        )
        engines = {'model': engine}
        return engines

    def _create_dataloader(
        self, dataset: MelDataset, training: bool
    ) -> DataLoader:
        collate_fn = None
        batch_size = (
            self.hparams.batch_size if training else self.hparams.valid_batch_size
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.hparams.loader_num_workers,
            persistent_workers=True,
            collate_fn=collate_fn,
        )

    def _build_dataset(self, training: bool) -> MelDataset:
        dataset: MelDataset
        data_dirs = (
            self.hparams.train_data_dirs if training else self.hparams.valid_data_dirs
        )
        dataset = MelDataset(self.hparams, data_dirs)
        return dataset

    def _save_checkpoint(self):
        self.hparams.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        for label, engine in self.engines.items():
            tag = f'{label}_{self.step}'
            engine.save_checkpoint(self.hparams.checkpoints_dir, tag=tag)
        log_info("Saved checkpoint after %d step", self.step)

    def _load_checkpoint(self):
        if not self.hparams.base_checkpoint:
            log_info("No checkpoint specified, nothing loaded")
            return

        self.step = self.hparams.base_checkpoint
        log_info("Loading checkpoint from %d step", self.step)

        for label, engine in self.engines.items():
            tag = f'{label}_{self.step}'
            engine.load_checkpoint(
                self.hparams.checkpoints_dir,
                tag=tag,
                load_module_strict=False,
                load_module_only=False,
            )
        self.step += 1

    def _make_infinite_epochs(self, dl: DataLoader):
        while True:
            yield from dl
            self.epoch += 1

    def _log_train_stats(self, stats):
        self._logger.update_running_vals(stats, 'training')
        self._logger.log_step(self.epoch, self.step, prefix='training')

        if self.step and self.step % self.hparams.steps_per_log == 0:
            self._logger.log_running_vals_to_tb(self.step)


if __name__ == "__main__":
    from torchsummary import summary

    mel_hparams = MelSpecVAEHParams()
    mel_spec_vae = MelSpecVAE(mel_hparams)
    summary(mel_spec_vae)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://{os.getenv("MASTER_ADDR")}:{os.getenv("MASTER_PORT")}',
        world_size=get_world_size(),
        rank=global_rank()

    )
    vae_supervisor = MelSpecVAESupervisor(mel_spec_vae, torch.device('cuda'), mel_hparams)
    vae_supervisor.train()
