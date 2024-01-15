"""
Our proposition of PerformanceRNN with self Attention on input mechanizm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from collections import namedtuple
import numpy as np
from progress.bar import Bar

from coopertunes.models import PerformanceRNN
from coopertunes.hparams.PerformanceRNN import PerformanceRNNHParams


class PerformanceRNNattentive(PerformanceRNN):
    def __init__(self, hparams: PerformanceRNNHParams, device: str = "cuda:0"):
        super().__init__(hparams, device)

        self.event_dim = hparams.event_dim
        self.control_dim = hparams.control_dim
        self.init_dim = hparams.init_dim
        self.hidden_dim = hparams.hidden_dim
        self.gru_layers = hparams.gru_layers
        self.concat_dim = hparams.event_dim + 1 + hparams.control_dim
        self.input_dim = hparams.hidden_dim
        self.output_dim = hparams.event_dim

        self.primary_event = self.event_dim - 1

        self.device = device

        self.inithid_fc = nn.Linear(
            hparams.init_dim, hparams.gru_layers * hparams.hidden_dim)
        self.inithid_fc_activation = nn.Tanh()

        self.event_embedding = nn.Embedding(
            hparams.event_dim, hparams.event_dim)
        self.concat_input_fc = nn.Linear(self.concat_dim, self.input_dim)
        self.concat_input_fc_activation = nn.LeakyReLU(0.1, inplace=True)

        self.gru = nn.GRU(self.input_dim, self.hidden_dim,
                          num_layers=hparams.gru_layers, dropout=hparams.gru_dropout)
        self.output_fc = nn.Linear(
            hparams.hidden_dim * hparams.gru_layers, self.output_dim)
        self.output_fc_activation = nn.Softmax(dim=-1)

        self.self_attention = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=8)
        self.attention_output = nn.LayerNorm(self.input_dim)



        self._initialize_weights()

    def forward(self, event, control=None, hidden=None):
        # One step forward

        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        event = self.event_embedding(event)

        if control is None:
            default = torch.ones(1, batch_size, 1).to(self.device)
            control = torch.zeros(
                1, batch_size, self.control_dim).to(self.device)
        else:
            default = torch.zeros(1, batch_size, 1).to(self.device)
            assert control.shape == (1, batch_size, self.control_dim)

        concat = torch.cat([event, default, control], -1)
        input = self.concat_input_fc(concat)
        input = self.concat_input_fc_activation(input)

        attention_output, _ = self.self_attention(input, input, input)
        attention_output += input
        attention_output = self.attention_output(attention_output)

        _, hidden = self.gru(attention_output, hidden)
        output = hidden.permute(1, 0, 2).contiguous()
        output = output.view(batch_size, -1).unsqueeze(0)
        output = self.output_fc(output)
        return output, hidden

    def generate(self, init, steps, events=None, controls=None, greedy=1.0,
                 temperature=1.0, teacher_forcing_ratio=1.0, output_type='index', verbose=False):
        # init [batch_size, init_dim]
        # events [steps, batch_size] indeces
        # controls [1 or steps, batch_size, control_dim]

        batch_size = init.shape[0]
        assert init.shape[1] == self.init_dim
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 2
            assert events.shape[0] >= steps - 1
            events = events[:steps-1]

        event = self.get_primary_event(batch_size)
        use_control = controls is not None
        if use_control:
            controls = self.expand_controls(controls, steps)
        hidden = self.init_to_hidden(init)

        outputs = []
        step_iter = range(steps)
        if verbose:
            step_iter = Bar('Generating').iter(step_iter)

        for step in step_iter:
            control = controls[step].unsqueeze(0) if use_control else None
            output, hidden = self.forward(event, control, hidden)

            use_greedy = np.random.random() < greedy
            event = self._sample_event(output, greedy=use_greedy,
                                       temperature=temperature)

            if output_type == 'index':
                outputs.append(event)
            elif output_type == 'softmax':
                outputs.append(self.output_fc_activation(output))
            elif output_type == 'logit':
                outputs.append(output)
            else:
                assert False

            if use_teacher_forcing and step < steps - 1:  # avoid last one
                if np.random.random() <= teacher_forcing_ratio:
                    event = events[step].unsqueeze(0)

        return torch.cat(outputs, 0)

    def beam_search(self, init, steps, beam_size, controls=None,
                    temperature=1.0, stochastic=False, verbose=False):
        raise NotImplementedError("PerformanceRNNattentive has no implemented beam search")