import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from ..hparams import MelSpecVAEHParams


class MelSpecVAE(nn.Module):
    """Generating mels from noise with vanilla VAE"""

    def __init__(self, hparams: MelSpecVAEHParams):
        super().__init__()
        self.hparams: MelSpecVAEHParams = hparams

        self.kld_weight = hparams.kld_weight
        self.latent_dim = hparams.latent_dim

        self.pool_factor = np.prod(hparams.conv_strides)

        self.before_latent = (
            hparams.input_shape[0]//self.pool_factor,
            hparams.input_shape[1]//self.pool_factor
        )
        self.last_filter = hparams.conv_filters[-1]

        self._build_encoder(
            hparams.conv_filters,
            hparams.conv_kernels,
            hparams.conv_strides,
            hparams.conv_padding
        )

        self.fc_mu = nn.Linear(
            int(self.last_filter*(self.before_latent[0]*self.before_latent[1])),
            self.latent_dim)

        self.fc_var = nn.Linear(
            int(self.last_filter*(self.before_latent[0]*self.before_latent[1])),
            self.latent_dim)

        self._build_decoder(
            hparams.conv_filters,
            hparams.conv_kernels,
            hparams.conv_strides,
            hparams.deconv_padding,
            hparams.deconv_out_padding
        )

    def _build_encoder(
        self,
        conv_filters: list,
        conv_kernels: list,
        conv_strides: list,
        conv_padding: list
    ):

        blocks = []
        in_channels = 1
        relu_fn = nn.ReLU()

        for (
                c_filter,
                c_kernel,
                c_stride,
                c_padd
            ) in zip(
                conv_filters,
                conv_kernels,
                conv_strides,
                conv_padding):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=c_filter,
                        kernel_size=c_kernel,
                        stride=c_stride,
                        padding=c_padd),
                    relu_fn,
                    nn.BatchNorm2d(c_filter),
                )
            )
            in_channels = c_filter

        self.encoder = nn.Sequential(*blocks)

    def _build_decoder(
        self,
        conv_filters: list,
        conv_kernels: list,
        conv_strides: list,
        deconv_padding: list,
        deconv_out_padding: list
    ):

        blocks = []

        decoder_input = nn.Linear(
            self.latent_dim,
            int(self.last_filter*(self.before_latent[0]*self.before_latent[1]))
        )

        relu_fn = nn.ReLU()
        for i in reversed(range(1, len(conv_filters))):
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=conv_filters[i],
                        out_channels=conv_filters[i-1],
                        kernel_size=conv_kernels[i],
                        stride=conv_strides[i],
                        padding=deconv_padding[i],
                        output_padding=deconv_out_padding[i]),
                    relu_fn,
                    nn.BatchNorm2d(conv_filters[i-1]),
                )
            )

        decoder_final_layer = nn.ConvTranspose2d(
            in_channels=conv_filters[0],
            out_channels=1,
            kernel_size=conv_kernels[0],
            stride=conv_strides[0],
            padding=deconv_padding[0],
            output_padding=deconv_out_padding[0]
        )

        self.decoder_input = decoder_input
        self.decoder = nn.Sequential(*blocks)
        self.final_layer = nn.Sequential(
            decoder_final_layer
        )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes
        """
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)

        return [mu, log_var]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        """
        result = self.decoder_input(z)
        result = rearrange(
            result,
            "b (lf bl_m bl_t) -> b lf bl_m bl_t",
            lf=self.last_filter,
            bl_m=self.before_latent[0],
            bl_t=self.before_latent[1]
        )
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def loss_function(
        self,
        y_recon: torch.Tensor,
        y_target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> dict:

        recons_loss = F.mse_loss(y_recon, y_target)
        kld_loss = torch.mean(
            -0.5 * torch.sum(
                1 + log_var - mu**2 - log_var.exp(), dim=1
            ), dim=0
        )

        loss = recons_loss + kld_loss * self.kld_weight
        return {"loss": loss, "recon": recons_loss.detach(), "kld": kld_loss.detach()}

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def inference(self, z: torch.Tensor):
        return self.decode(z)
