from pathlib import Path
import time

import librosa
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

        self.netG = generator.to(self.device)
        self.netD = discriminator.to(self.device)
        self.audio2mel = audio2mel.to(self.device)

        self.hparams = hparams

        self.epoch = 1
        self.step = 1

        self._logger = Logger("melgan", self.hparams, device)

        self.train_dl, self.val_dl = self._build_loaders()

        self.optG = torch.optim.Adam(
            self.netG.parameters(), lr=hparams.learning_rate, betas=hparams.adam_betas)
        self.optD = torch.optim.Adam(
            self.netD.parameters(), lr=hparams.learning_rate, betas=hparams.adam_betas)

        if self.hparams.base_checkpoint:
            self._load_checkpoint()
        else:
            log_info("Initilizing fresh training")

    def train(self):
        start = time.time()

        torch.backends.cudnn.benchmark = True

        while True:
            if self.step >= self.hparams.total_steps:
                log_info("Max steps reached. Training finished")
                break

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()

            start = time.time()

            for _, x_t in enumerate(self.train_dl):
                x_t = x_t.to(self.device)
                s_t = self.audio2mel(x_t).detach()
                x_pred_t = self.netG(s_t.to(self.device))

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

                stats = {
                    'discriminator': loss_D.item(),
                    'generator': loss_G.item(),
                    'feature_matching': loss_feat.item(),
                    'mel_reconstruction': s_error,
                    'step_time': (time.time() - start),
                }
                self._log_train_stats(stats)

                if self.step % self.hparams.steps_per_ckpt == 0:
                    self._save_checkpoint()
                    self.netG.eval()
                    self.netD.eval()
                    self.eval(s_error)
                    self.netG.train()
                    self.netD.train()
                self.step += 1

    @torch.inference_mode()
    def eval(self, mel_recon):
        best_mel_reconst = 1000000
        for i, x_t in enumerate(self.val_dl):
            x_t = x_t.to(self.device)
            s_t = self.audio2mel(x_t).detach()
            pred_audio = self.netG(s_t.to(self.device))
            pred_audio = pred_audio.squeeze().cpu()
            save_sample(
                self.hparams.logs_dir / (f"generated_{i}.wav"),
                self.hparams.sampling_rate,
                pred_audio
            )
            self._logger.log_audio(
                f"generated/sample_{i}.wav",
                pred_audio,
                self.epoch,
                sample_rate=22050,
            )

        if new_best := mel_recon < best_mel_reconst:
            best_mel_reconst = mel_recon

        self._save_checkpoint(best=new_best)

    def _save_checkpoint(self, best: bool = False):
        self.hparams.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_state = {
            "step": self.step,
            "epoch": self.epoch,
            "netG": self.netG.state_dict(),
            "optG": self.optG.state_dict(),
            "netD": self.netD.state_dict(),
            "optD": self.optD.state_dict(),
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
        audio_tensor = torch.from_numpy(audio)[None]
        spec = self.audio2mel(audio_tensor.unsqueeze(1).to(self.device))
        reconstructed = self.netG(spec.to(self.device)).squeeze((0, 1)).detach().cpu().numpy()
        sf.write(output_path, reconstructed, sr)

    def __call__(self, spectrogram: np.ndarray):
        """
        Converts spectrogram to raw audio.
        spectrogram's shape is [1, bins, len]
        """
        return self.netG(
            torch.from_numpy(spectrogram).float().to(self.device)
        ).squeeze(1)

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_dataset = self._build_dataset(training=True)
        train_dl = self._create_dataloader(train_dataset, training=True)

        val_dataset = self._build_dataset(training=False)
        val_dl = self._create_dataloader(val_dataset, training=False)

        test_voc = []
        test_audio = []
        for i, x_t in enumerate(val_dl):
            x_t = x_t.cuda()
            s_t = self.audio2mel(x_t).detach()

            test_voc.append(s_t.cuda())
            test_audio.append(x_t)

            audio = x_t.squeeze().cpu()
            save_sample(
                self.hparams.logs_dir / f"original_{i}.wav",
                self.hparams.sampling_rate,
                audio
            )
            self._logger.log_audio(
                f"original/sample_{i}.wav", audio, 0, sample_rate=self.hparams.sampling_rate)

            if i == self.hparams.n_test_samples - 1:
                break

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

    def load_pretrained(self):
        checkpoint = torch.load(self.hparams.default_checkpoint)
        log_info("Loading checkpoint from pretrained from authors", checkpoint["step"])

        self.netG.load_state_dict(checkpoint["netG"])
        self.optG.load_state_dict(checkpoint["optG"])
        self.netD.load_state_dict(checkpoint["netD"])
        self.optD.load_state_dict(checkpoint["optD"])
        self.step = checkpoint["step"]
        self.step += 1
        self.epoch = checkpoint["epoch"]

if __name__ == "__main__":
    mel_hparams = MelGanHParams()
    audio2mel_hparams = Audio2MelHParams()

    melGanAudio2mel = Audio2Mel(audio2mel_hparams)
    melGanGgenerator = MelGanGenerator(mel_hparams)
    melGanDiscriminator = MelGanDiscriminator(mel_hparams)

    supervisor = MelGanSupervisor(
        melGanGgenerator,
        melGanDiscriminator,
        melGanAudio2mel,
        get_default_device(),
        mel_hparams
    )
    supervisor.train()
