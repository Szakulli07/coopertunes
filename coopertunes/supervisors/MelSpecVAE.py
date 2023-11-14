from ..models import MelSpecVAE
from ..hparams import MelSpecVAEHParams

from ..utils import log_info

import torch

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

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

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

            batch = next(train_dl_iter)
            mels = convert_audios2mels(batch['audio'], 16000).to(device)
            print(mels.shape)

            self.optimizer.zero_grad()
            reconstruct, input, mu, log_var = self.model(mels)
            loss = self.model.loss_function(reconstruct, input, mu, log_var)

            loss['loss'].backward()

            log_info(f"step: {self.step} | loss: {loss['loss'].clone().detach().item()} | recon: {loss['Reconstruction_Loss']} | kld: {loss['KLD']}")
            self.optimizer.step()

            self.step += 1

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_dataset = self._build_dataset(training=True)
        train_dl = self._create_dataloader(train_dataset, training=True)

        val_dataset = self._build_dataset(training=False)
        val_dl = self._create_dataloader(val_dataset, training=False)

        return train_dl, val_dl

    def _create_dataloader(
        self, dataset: AudioDataset, training: bool
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

    def _build_dataset(self, training: bool) -> AudioDataset:
        dataset: AudioDataset
        data_dirs = (
            self.hparams.train_data_dirs if training else self.hparams.valid_data_dirs
        )
        dataset = AudioDataset(self.hparams, data_dirs)
        return dataset

    def _make_infinite_epochs(self, dl: DataLoader):
        while True:
            yield from dl
            self.epoch += 1
