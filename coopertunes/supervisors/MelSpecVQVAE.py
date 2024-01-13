import time
from statistics import mean

import torch
from einops import rearrange
from torch.utils.data import DataLoader

from ..datasets import MelDataset
from ..hparams import MelSpecVQVAEHParams
from ..logger import Logger
from ..models import MelSpecVQVAE
from ..utils import log_info


class MelSpecVQVAESupervisor:
    """Supervisor for MelSpecVQVAESupervisor
    After init you can launch training with `train` method"""

    def __init__(self, model: MelSpecVQVAE, device: torch.device, hparmas: MelSpecVQVAEHParams):
        self.model = model
        self.device = device
        self.hparams = hparmas

        self.epoch = 1
        self.step = 1

        self._logger = Logger("melspecvqvae", self.hparams, device)

        self.train_dl, self.val_dl = self._build_loaders()

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=hparmas.lr, betas=hparams.betas)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.hparams.lr_decay
        )

        if self.hparams.base_checkpoint:
            self._load_checkpoint()
        else:
            log_info("Initilizing fresh training")

    def train(self):
        """train main function."""
        log_info("Started training")
        train_dl_iter = self._make_infinite_epochs(self.train_dl)
        self.model.to(self.device)
        self.model.train()

        # Training loop
        while True:
            # break training if reached total_steps
            if self.step >= self.hparams.total_steps:
                log_info("Max steps reached. Training finished")
                break

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()

            start = time.time()

            self.optimizer.zero_grad()
            for _ in range(hparams.grad_accumulation_steps):
                batch = next(train_dl_iter)
                mels = batch["mels"].to(self.device)

                reconstruct, x, vq_loss = self.model(mels)
                loss = self.model.loss_function(reconstruct, x, vq_loss)
                loss["loss"].backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
               self.model.parameters(), hparams.grad_clip_thresh
            )

            self.optimizer.step()
            self.scheduler.step()

            stats = {
                'loss': loss["loss"].item(),
                'recon': loss["recon"].item(),
                'vq': loss["vq"].item(),
                'grad_norm': grad_norm.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'step_time': (time.time() - start),
            }
            self._log_train_stats(stats)

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()
                self.model.eval()
                self.eval()
                self.model.train()

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
            reconstruct, x, vq_loss = self.model(mels)
            loss = self.model.loss_function(reconstruct, x, vq_loss)

            loss_list.append(loss["loss"].item())
            recon_list.append(loss["recon"].item())
            kld_list.append(loss["vq"].item())

            if i == 0:
                reconstructs = [None] * batch["mels"].shape[0]
                for i, recon in enumerate(rearrange(reconstruct, 'n 1 c t ->n c t')):
                    reconstructs[i] = recon.detach()

        stats = {
            'loss': mean(loss_list),
            'recon': mean(recon_list),
            'vq': mean(kld_list),
            'step_time': (time.time() - start),
        }

        self._logger.log_audio(batch=reconstructs, step=self.step, audio_type='output')
        self._logger.update_running_vals(stats, 'validation')
        self._logger.log_step(self.epoch, self.step, prefix='validation')
        self._logger.log_running_vals_to_tb(self.step)
        log_info('Validation ends')

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_dataset = self._build_dataset(training=True)
        train_dl = self._create_dataloader(train_dataset, training=True)

        val_dataset = self._build_dataset(training=False)
        val_dl = self._create_dataloader(val_dataset, training=False)

        for i, batch in enumerate(val_dl):
            if i == 0:
                mels = [None] * batch["mels"].shape[0]
                for j, mel in enumerate(rearrange(batch["mels"], 'n 1 c t ->n c t')):
                    mels[j] = mel

        self._logger.log_audio(mels, step=0, audio_type='target')

        return train_dl, val_dl

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
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            }, (self.hparams.checkpoints_dir/str(self.step)).with_suffix(".pt")
        )
        log_info("Saved checkpoint after %d step", self.step)

    def _load_checkpoint(self):
        if not self.hparams.base_checkpoint:
            log_info("No checkpoint specified, nothing loaded")
            return

        checkpoint = torch.load(self.hparams.base_checkpoint)
        log_info("Loading checkpoint from %d step", self.step)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
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

    hparams = MelSpecVQVAEHParams()
    mel_spec_vae = MelSpecVQVAE(hparams)
    summary(mel_spec_vae)
    cpu_device = torch.device("cpu")
    vae_supervisor = MelSpecVQVAESupervisor(mel_spec_vae, cpu_device, hparams)
    vae_supervisor.train()
