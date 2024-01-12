import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from ..hparams import HParams


class FrameRNN(nn.Module):

    def __init__(self, frame_size, num_lower_tier_frames,
                 num_layers, dim, q_levels, dropout):
        super().__init__()
        self.frame_size = frame_size
        self.num_lower_tier_frames = num_lower_tier_frames
        self.num_layers = num_layers
        self.dim = dim
        self.q_levels = q_levels

        h0 = torch.zeros(num_layers, dim)
        self.h0 = torch.nn.Parameter(h0)

        self.inputs = nn.Linear(
            in_features=self.frame_size,
            out_features=self.dim
        )
        self.rnn = nn.GRU(
            input_size=self.dim,
            hidden_size=self.dim,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=dropout
        )
        self.conv_t = nn.ConvTranspose1d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.frame_size,
            stride=self.frame_size,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_frames: torch.Tensor = None,
        hidden: torch.Tensor = None
    ):
        batch_size = x.shape[0]

        x_frames = rearrange(x, 'b (f fs) 1 -> b f fs', fs=self.frame_size)
        x_frames = ((x_frames / (self.q_levels / 2.0)) - 1.0) * 2.0
        num_steps = x_frames.shape[1]

        x_hat_frames = self.inputs(x_frames)

        if cond_frames is not None:
            x_hat_frames += cond_frames

        if hidden is None:
            hidden = repeat(self.h0, 'l d -> l b d', b=batch_size)

        x_hat_frames, hidden = self.rnn(x_hat_frames, hidden)

        x_hat = self.conv_t(rearrange(x_hat_frames, 'b ltf d -> b d ltf'))
        x_hat = rearrange(x_hat, 'b c d -> b d c')
        return x_hat, hidden


class SampleMLP(nn.Module):

    def __init__(self, frame_size, dim, q_levels):
        super().__init__()
        self.dim = dim
        self.q_levels = q_levels
        self.embedding = nn.Embedding(
            self.q_levels,
            self.q_levels
        )
        self.conv = nn.Conv1d(
            in_channels=self.q_levels,
            out_channels=self.dim,
            kernel_size=frame_size,
            bias=False
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.dim,
                out_features=self.dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.dim,
                out_features=self.dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.dim,
                out_features=self.q_levels
            )
        )

    def forward(self, x: torch.Tensor, cond_frames: torch.Tensor):
        batch_size = x.shape[0]
        x_hat = self.embedding(rearrange(x, 'b t 1 -> (b t 1)'))
        x_hat = rearrange(x_hat, '(b t) c -> b c t', b=batch_size)
        x_hat = self.conv(x_hat)

        return self.mlp(x_hat + cond_frames)


class SampleRNN(nn.Module):
    """Generating audio from noise with hierarchical RNN"""

    def __init__(self, batch_size, frame_sizes, q_levels, q_type, dim, rnn_type,
                 num_rnn_layers, seq_len, emb_size, skip_conn, rnn_dropout):
        super().__init__()
        self.batch_size = batch_size
        self.big_frame_size = frame_sizes[1]
        self.frame_size = frame_sizes[0]
        self.q_type = q_type
        self.q_levels = q_levels
        self.dim = dim
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers

        self.big_frame_rnn = FrameRNN(
            rnn_type=self.rnn_type,
            frame_size=self.big_frame_size,
            num_lower_tier_frames=self.big_frame_size // self.frame_size,
            num_layers=self.num_rnn_layers,
            dim=self.dim,
            q_levels=self.q_levels,
            dropout=rnn_dropout
        )

        self.frame_rnn = FrameRNN(
            rnn_type=self.rnn_type,
            frame_size=self.frame_size,
            num_lower_tier_frames=self.frame_size,
            num_layers=self.num_rnn_layers,
            dim=self.dim,
            q_levels=self.q_levels,
            dropout=rnn_dropout
        )

        self.sample_mlp = SampleMLP(
            self.frame_size,
            self.dim,
            self.q_levels
        )

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def inference(self, z: torch.Tensor):
        return self.decode(z)
