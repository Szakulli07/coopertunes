import os
import time
from statistics import mean

import numpy as np

import torch
from torch import nn
from torch import optim
from einops import rearrange
from torch.utils.data import DataLoader

from coopertunes.datasets import MidiDataset

from coopertunes.hparams import PerformanceRNNHParams
from coopertunes.logger import Logger
from coopertunes.models import PerformanceRNN
from coopertunes.utils import log_info, transposition, compute_gradient_norm, find_files_by_extensions, event_indeces_to_midi_file

from coopertunes.datatools.miditools import NoteSeq, EventSeq, ControlSeq, Control


class PerformanceRNNSupervisor:
    """Supervisor for PerformanceRNNSupervisor
    After init you can launch training with `train` method
    You can generate sample using "generate" method."""

    def __init__(self, model: PerformanceRNN, device: torch.device, hparams: PerformanceRNNHParams):
        self.hparams = hparams
        self.sess_path = hparams.logs_dir
        self.data_path = hparams.train_data_dirs[0]
        self.saving_interval = hparams.steps_per_ckpt

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.window_size = hparams.window_size
        self.stride_size = hparams.stride_size
        self.use_transposition = hparams.use_transposition
        self.control_ratio = hparams.control_ratio
        self.teacher_forcing_ratio = hparams.teacher_forcing_ratio
        self.reset_optimizer = hparams.reset_optimizer
        self.enable_logging = hparams.enable_logging

        self.event_dim = EventSeq.dim()
        self.control_dim = ControlSeq.dim()

        self.device = device

        self.step = 1

        self.model = model
        self.optimizer = optim.Adam(
            model.parameters(), lr=hparams.learning_rate)
        self._logger = Logger("performancernn", hparams, device)

        self.train_dl, self.val_dl = self._build_loaders()

        model.to(self.device)

        if self.hparams.base_checkpoint:
            self._load_checkpoint()
        else:
            log_info("Initilizing fresh training")

    def train(self):
        self.model.train()
        start = time.time()
        loss_function = nn.CrossEntropyLoss()

        for iteration, (events, controls) in enumerate(self.train_dl):
            if self.step >= self.hparams.total_steps:
                log_info("Max steps reached. Training finished")
                break

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()

            if self.use_transposition:
                offset = np.random.choice(np.arange(-6, 6))
                events, controls = transposition(events, controls, offset)

            events = torch.LongTensor(events).to(self.device)
            assert events.shape[0] == self.window_size

            if np.random.random() < self.control_ratio:
                controls = torch.FloatTensor(controls).to(self.device)
                assert controls.shape[0] == self.window_size
            else:
                controls = None

            init = torch.randn(
                self.batch_size, self.model.init_dim).to(self.device)
            outputs = self.model.generate(init, self.window_size, events=events[:-1], controls=controls,
                                          teacher_forcing_ratio=self.teacher_forcing_ratio, output_type='logit')
            assert outputs.shape[:2] == events.shape[:2]

            loss = loss_function(
                outputs.view(-1, self.event_dim), events.view(-1))
            self.model.zero_grad()
            loss.backward()

            norm = compute_gradient_norm(self.model.parameters())
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            stats = {
                'generator': loss.item(),
                'gradient_norm': norm.item(),
                'step_time': (time.time() - start),
            }
            self._log_train_stats(stats)

            self.step += 1

    def generate(self, output_dir, control=None, init_zero=False, use_beam_search=False):
        """
        Generate music sample.
        It uses options specified at hparams, and:
        output dir: where to save generated samples.
        control: key of generated audio. example C dur is '1,0,1,0,1,1,0,1,0,1,0,1;3'. 
            May be None, or path to Midi file for guidance.
        """

        if control is not None:
            if os.path.isfile(control) or os.path.isdir(control):
                if os.path.isdir(control):
                    files = list(find_files_by_extensions(control))
                    assert len(files) > 0, f'no file in "{control}"'
                    control = np.random.choice(files)
                _, compressed_controls = torch.load(control)
                controls = ControlSeq.recover_compressed_array(
                    compressed_controls)
                if max_len == 0:
                    max_len = controls.shape[0]
                controls = torch.tensor(controls, dtype=torch.float32)
                controls = controls.unsqueeze(1).repeat(
                    1, self.batch_size, 1).to(device)
                control = f'control sequence from "{control}"'

            else:
                pitch_histogram, note_density = control.split(';')
                pitch_histogram = list(filter(len, pitch_histogram.split(',')))
                if len(pitch_histogram) == 0:
                    pitch_histogram = np.ones(12) / 12
                else:
                    pitch_histogram = np.array(
                        list(map(float, pitch_histogram)))
                    assert pitch_histogram.size == 12
                    assert np.all(pitch_histogram >= 0)
                    pitch_histogram = pitch_histogram / pitch_histogram.sum() \
                        if pitch_histogram.sum() else np.ones(12) / 12
                note_density = int(note_density)
                assert note_density in range(len(ControlSeq.note_density_bins))
                control = Control(pitch_histogram, note_density)
                controls = torch.tensor(
                    control.to_array(), dtype=torch.float32)
                controls = controls.repeat(1, self.batch_size, 1).to(device)
                control = repr(control)
        else:
            controls = None
            control = 'NONE'

        use_beam_search = self.hparams.beam_size > 0
        greedy_ratio = self.hparams.greedy_ratio
        beam_size = self.hparams.beam_size
        if use_beam_search:
            greedy_ratio = 'DISABLED'
        else:
            beam_size = 'DISABLED'
        self.model.eval()
        if init_zero:
            init = torch.zeros(self.batch_size, model.init_dim).to(device)
        else:
            init = torch.randn(self.batch_size, model.init_dim).to(device)

        with torch.no_grad():
            if use_beam_search:
                outputs = model.beam_search(init, self.hparams.max_len, beam_size,
                                            controls=controls,
                                            temperature=self.hparams.temperature,
                                            stochastic=self.hparams.stochastic_beam_search,
                                            verbose=True)
            else:
                outputs = model.generate(init, self.hparams.max_len,
                                         controls=controls,
                                         greedy=greedy_ratio,
                                         temperature=self.hparams.temperature,
                                         verbose=True)

        outputs = outputs.cpu().numpy().T  # [batch, steps]

        os.makedirs(output_dir, exist_ok=True)
        for i, output in enumerate(outputs):
            name = f'output-{i:03d}.mid'
            path = os.path.join(output_dir, name)
            event_indeces_to_midi_file(output, path)

    def _build_dataset(self):
        dataset = MidiDataset(self.data_path, verbose=True)
        dataset_size = len(dataset.samples)
        assert dataset_size > 0
        return dataset

    def _build_loaders(self) -> tuple[DataLoader, DataLoader | None]:
        dataset = self._build_dataset()
        batch_gen = dataset.batches(
            self.batch_size, self.window_size, self.stride_size)
        return batch_gen, None

    def _save_checkpoint(self, best: bool = False):
        self.hparams.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()}
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

        path = os.path.join(self.hparams.checkpoints_dir,
                            str(self.hparams.base_checkpoint)+".pt")
        checkpoint = torch.load(path)
        log_info("Loading checkpoint from %d step", checkpoint["step"])

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        self.step += 1

    def _log_train_stats(self, stats):
        self._logger.update_running_vals(stats, 'training')
        self._logger.log_step(self.step, prefix='training')

        if self.step and self.step % self.hparams.steps_per_log == 0:
            self._logger.log_running_vals_to_tb(self.step)

    def load_pretrained(self):
        checkpoint = torch.load(self.hparams.default_checkpoint)
        log_info("Loading checkpoint from pretrained from authors")

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        self.step += 1


if __name__ == "__main__":
    hparams = PerformanceRNNHParams()
    model = PerformanceRNN(hparams)
    device = "cuda:0"
    supervisor = PerformanceRNNSupervisor(model, device, hparams)

    # LOAD PRETRAINED WEIGHTS FROM AUTHORS
    # supervisor.load_pretrained()

    # TRAIN MODEL
    supervisor.train()

    # GENERATE SAMPLES
    # key = '1,0,1,0,1,1,0,1,0,1,0,1;3'
    # supervisor.generate("performance_rnn_results", control=key)
