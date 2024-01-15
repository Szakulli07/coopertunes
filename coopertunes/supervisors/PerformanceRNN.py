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
from coopertunes.utils import log_info, transposition, compute_gradient_norm

from coopertunes.datatools.miditools import NoteSeq, EventSeq, ControlSeq


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
        self.epoch = 1

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
        self.logger = Logger("performancernn", hparams, device)

        model.to(self.device)
        

    def train(self):
        dataset = self.load_dataset()
        if self.enable_logging:
            self.writer = self.logger.get_summary_writer()

    
        loss_function = nn.CrossEntropyLoss()

        batch_gen = dataset.batches(self.batch_size, self.window_size, self.stride_size)

        for iteration, (events, controls) in enumerate(batch_gen):
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

            init = torch.randn(self.batch_size, self.model.init_dim).to(self.device)
            outputs = self.model.generate(init, self.window_size, events=events[:-1], controls=controls,
                                    teacher_forcing_ratio=self.teacher_forcing_ratio, output_type='logit')
            assert outputs.shape[:2] == events.shape[:2]

            loss = loss_function(outputs.view(-1, self.event_dim), events.view(-1))
            self.model.zero_grad()
            loss.backward()

            norm = compute_gradient_norm(self.model.parameters())
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()

            if self.enable_logging:
                self.writer.add_scalar('model/loss', loss.item(), iteration)
                self.writer.add_scalar('model/norm', norm.item(), iteration)

            print(f'iter {iteration}, loss: {loss.item()}')

    def generate(self):
        pass

    def load_dataset(self):
        dataset = MidiDataset(self.data_path, verbose=True)
        dataset_size = len(dataset.samples)
        assert dataset_size > 0
        return dataset    

    def _save_checkpoint(self, best: bool = False):
        self.hparams.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_state = {
            "step": self.step,
            "epoch": self.epoch,
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

        checkpoint = torch.load(self.hparams.base_checkpoint)
        log_info("Loading checkpoint from %d step", checkpoint["step"])

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"]
        self.step += 1
        self.epoch = checkpoint["epoch"]


if __name__ == "__main__":
    hparams = PerformanceRNNHParams()
    model = PerformanceRNN(hparams)
    device = "cuda:0"
    supervisor = PerformanceRNNSupervisor(model, device, hparams)

    supervisor.train()