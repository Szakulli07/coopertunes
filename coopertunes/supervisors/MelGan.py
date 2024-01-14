import os
import time

import torch
import librosa
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader
from pathlib import Path

from coopertunes.datasets import AudioDataset
from coopertunes.hparams import MelGanHParams, Audio2MelHParams
from coopertunes.logger import Logger
from coopertunes.models import MelGanGenerator, MelGanDiscriminator, Audio2Mel
from coopertunes.utils import (
    save_sample,
    get_default_device,
    log_info
)

class MelGanSupervisor:
    """Supervisor for MelGAN
    After init you can launch training with `train` method
    You can test trained checkpoints with `test` method on given raw audio
    """

    def __init__(
        self,
        generator: MelGanGenerator,
        discriminator: MelGanDiscriminator,
        audio2mel: Audio2Mel,
        device: torch.device,
        hparams: MelGanHParams,
    ):
        self.device = device

        self.netG = generator
        self.netD = discriminator
        self.audio2mel = audio2mel

        self.hparams = hparams

        self.epoch = 1
        self.step = 1

        self._logger = Logger("melgan", self.hparams, device)

        self.train_dl, self.val_dl = self._build_loaders()

        self.optG = torch.optim.Adam(
            self.netG.parameters(), lr=hparams.learning_rate, betas=hparams.adam_betas)
        self.optD = torch.optim.Adam(
            self.netD.parameters(), lr=hparams.learning_rate, betas=hparams.adam_betas)


        self.writer = self._logger.get_summary_writer()

        if self.hparams.base_checkpoint:
            self._load_checkpoint()
        else:
            log_info("Initilizing fresh training")


    def train(self):
        root = Path(self.hparams.summary_path)
        root.mkdir(parents=True, exist_ok=True)

        train_loader, test_loader = self._create_dataloaders()

        test_voc = []
        test_audio = []
        for i, x_t in enumerate(test_loader):
            x_t = x_t.cuda()
            s_t = self.audio2mel(x_t).detach()

            test_voc.append(s_t.cuda())
            test_audio.append(x_t)

            audio = x_t.squeeze().cpu()
            save_sample(root / ("original_%d.wav" % i), self.hparams.sampling_rate, audio)
            self.writer.add_audio(
                "original/sample_%d.wav" % i, audio, 0, sample_rate=self.hparams.sampling_rate)

            if i == self.hparams.n_test_samples - 1:
                break

        costs = []
        start = time.time()

        torch.backends.cudnn.benchmark = True

        steps = 0
        for epoch in range(1, self.hparams.epochs + 1):
            for iterno, x_t in enumerate(train_loader):
                x_t = x_t.cuda()
                s_t = self.audio2mel(x_t).detach()
                x_pred_t = self.netG(s_t.cuda())

                with torch.no_grad():
                    s_pred_t = self.audio2mel(x_pred_t.detach())
                    s_error = F.l1_loss(s_t, s_pred_t).item()

                D_fake_det = self.netD(x_pred_t.cuda().detach())
                D_real = self.netD(x_t.cuda())

                loss_D = 0
                for scale in D_fake_det:
                    loss_D += F.relu(1 + scale[-1]).mean()

                for scale in D_real:
                    loss_D += F.relu(1 - scale[-1]).mean()

                self.netD.zero_grad()
                loss_D.backward()
                self.optD.step()

                D_fake = self.netD(x_pred_t.cuda())

                loss_G = 0
                for scale in D_fake:
                    loss_G += -scale[-1].mean()

                loss_feat = 0
                feat_weights = 4.0 / (self.hparams.n_layers_D + 1)
                D_weights = 1.0 / self.hparams.num_D
                wt = D_weights * feat_weights
                for i in range(self.hparams.num_D):
                    for j in range(len(D_fake[i]) - 1):
                        loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

                self.netG.zero_grad()
                (loss_G + self.hparams.lambda_feat * loss_feat).backward()
                self.optG.step()

                costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])

                self.writer.add_scalar("loss/discriminator", costs[-1][0], steps)
                self.writer.add_scalar("loss/generator", costs[-1][1], steps)
                self.writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
                self.writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
                steps += 1

                if steps % self.hparams.save_interval == 0:
                    self._eval(root, test_voc, test_audio, epoch, costs)

                if steps % self.hparams.log_interval == 0:
                    print(
                        "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                            epoch,
                            iterno,
                            len(train_loader),
                            1000 * (time.time() - start) / self.hparams.log_interval,
                            np.asarray(costs).mean(0),
                        )
                    )
                    costs = []
                    start = time.time()

    def _eval(self, root, test_voc, test_audio, epoch, costs):
        best_mel_reconst = 1000000
        with torch.no_grad():
            for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                pred_audio = self.netG(voc)
                pred_audio = pred_audio.squeeze().cpu()
                save_sample(root / ("generated_%d.wav" % i), hparams.sampling_rate, pred_audio)
                self.writer.add_audio(
                    "generated/sample_%d.wav" % i,
                    pred_audio,
                    epoch,
                    sample_rate=22050,
                )

        if new_best := np.asarray(costs).mean(0)[-1] < best_mel_reconst:
            best_mel_reconst = np.asarray(costs).mean(0)[-1]

        self._save_checkpoint(best = new_best)


    def _save_checkpoint(self, best: bool = False):
        self.hparams.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_state = {
            "step": self.step,
            "epoch": self.epoch,
            "netG": self.netG.state_dict(),
            "optG": self.optG.state_dict(),
            "netD": self.netD.state_dict(),
            "optD": self.optD.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(
            checkpoint_state,
            (self.hparams.checkpoints_dir/str(self.step)).with_suffix(".pt")
        )

        if best:
            torch.save(
                checkpoint_state,
                (self.hparams.checkpoints_dir/'best').with_suffix(".pt")
            )
        log_info("Saved checkpoint after %d step", self.step)


    def _load_checkpoint(self):
        if not self.hparams.base_checkpoint:
            log_info("No checkpoint specified, nothing loaded")
            return

        checkpoint = torch.load(self.hparams.base_checkpoint)
        log_info("Loading checkpoint from %d step", checkpoint["step"])

        self.netG.load_state_dict(checkpoint["netG"])
        self.optG.load_state_dict(checkpoint["optG"])
        self.netD.load_state_dict(checkpoint["netD"])
        self.optD.load_state_dict(checkpoint["optD"])
        self.step = checkpoint["step"]
        self.step += 1
        self.epoch = checkpoint["epoch"]

    def test(self, audio_path: str, output_path: str = "melgan_result.wav"):
        """
        It allows to reconstruct given raw audio using currently loaded generator.
        Audio will be converted to Mel Spectrogram, then back to raw audio, and saved.
        """
        audio, sr = librosa.core.load(audio_path)
        audio = torch.from_numpy(audio)[None]
        spec = self.audio2mel(audio.unsqueeze(1).to(self.device))
        reconstructed = self.netG(spec.to(self.device)).squeeze((0,1)).detach().cpu().numpy()
        sf.write(output_path, reconstructed, sr)


    def __call__(self, spectrogram: np.array):
        """
        Converts spectrogram to raw audio. 
        spectrogram's shape is [1, bins, len]
        """
        return self.netG(spectrogram.to(self.device)).squeeze(1)


    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_dataset = self._build_dataset(training=True)
        train_dl = self._create_dataloader(train_dataset, training=True)

        val_dataset = self._build_dataset(training=False)
        val_dl = self._create_dataloader(val_dataset, training=False)

        return train_dl, val_dl

    def _create_dataloader(
        self, dataset: AudioDataset, training: bool
    ) -> DataLoader:
        dataloader: DataLoader
        if training:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.hparams.batch_size,
                num_workers=0
            )
        else:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1
            )
        return dataloader

    def _build_dataset(self, training: bool) -> AudioDataset:
        dataset: AudioDataset
        if training:
            dataset = AudioDataset(
                training_files=Path(self.hparams.processed_data_dir/"train_files.txt"),
                segment_length=self.hparams.seq_len,
                sampling_rate=self.hparams.sampling_rate
            )
        else:
            dataset = AudioDataset(
                training_files=Path(self.hparams.processed_data_dir/"test_files.txt"),
                segment_length=self.hparams.sampling_rate * 4,
                sampling_rate=self.hparams.sampling_rate,
                augment=False,
            )
        return dataset

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
    hparams = MelGanHParams()
    audio2mel_hparams = Audio2MelHParams()

    audio2mel = Audio2Mel(audio2mel_hparams)
    generator = MelGanGenerator(hparams)
    discriminator = MelGanDiscriminator(hparams)

    supervisor = MelGanSupervisor(
        generator,
        discriminator,
        audio2mel,
        get_default_device(),
        hparams
    )
    supervisor.train()
