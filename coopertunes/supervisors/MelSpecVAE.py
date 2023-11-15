import torch
from torch.utils.data import DataLoader

from ..datasets import MelDataset
from ..hparams import MelSpecVAEHParams
from ..models import MelSpecVAE
from ..utils import log_info


class MelSpecVAESupervisor:
    '''Common supervisor for coopertunes models
    After init you can launch training with `train` method'''

    def __init__(self, model: MelSpecVAE, device: torch.device, hparmas: MelSpecVAEHParams):
        self.model = model
        self.device = device
        self.hparams = hparmas

        self.epoch = 1
        self.step = 1

        self.train_dl, self.val_dl = self._build_loaders()

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=hparmas.learning_rate)

        if self.hparams.base_checkpoint:
            self._load_checkpoint()
        else:
            log_info('Initilizing fresh training')

    def train(self):
        '''train main function.'''
        log_info('Started training')
        train_dl_iter = self._make_infinite_epochs(self.train_dl)
        self.model.to(self.device)
        self.model.train()

        # Training loop
        while True:
            # break training if reached total_steps
            if self.step >= self.hparams.total_steps:
                log_info('Max steps reached. Training finished')
                break

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()

            batch = next(train_dl_iter)
            mels = batch['mels'].to(self.device)

            self.optimizer.zero_grad()
            reconstruct, input, mu, log_var = self.model(mels)
            loss = self.model.loss_function(reconstruct, input, mu, log_var)

            loss['loss'].backward()

            log_info(f"step: {self.step} | loss: {loss['loss'].clone().detach().item()} | recon: {loss['recon']} | kld: {loss['kld']}")
            self.optimizer.step()

            self.step += 1

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_dataset = self._build_dataset(training=True)
        train_dl = self._create_dataloader(train_dataset, training=True)

        val_dataset = self._build_dataset(training=False)
        val_dl = self._create_dataloader(val_dataset, training=False)

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
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, (self.hparams.checkpoints_dir/str(self.step)).with_suffix('.pt')
        )
        log_info('Saved checkpoint after %d step', self.step)

    def _load_checkpoint(self):
        if not self.hparams.base_checkpoint:
            log_info('No checkpoint specified, nothing loaded')
            return

        checkpoint = torch.load(self.hparams.base_checkpoint)
        log_info('Loading checkpoint from %d step', self.step)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.step += 1

    def _make_infinite_epochs(self, dl: DataLoader):
        while True:
            yield from dl
            self.epoch += 1


if __name__ == "__main__":
    hparams = MelSpecVAEHParams()
    model = MelSpecVAE(hparams)
    device = 'cpu'
    supervisor = MelSpecVAESupervisor(model, device, hparams)
    supervisor.train()
