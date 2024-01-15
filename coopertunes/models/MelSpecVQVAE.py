from typing import Any

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from coopertunes.hparams import MelSpecVQVAEHParams


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = rearrange(latents, 'b d h w -> b h w d')
        flat_latents = rearrange(latents, 'b h w d -> (b h w) d')

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(flat_latents,
                             self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1)
        encoding_inds = rearrange(encoding_inds, '... -> ... 1')

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(
            encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(
            encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = rearrange(
            quantized_latents,
            '(b h w) d -> b h w d',
            b=latents.shape[0],
            h=latents.shape[1],
            w=latents.shape[2],
        )

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # [B x D x H x W]
        return rearrange(quantized_latents, 'b h w d -> b d h w'), vq_loss


class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.ReLU(True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.resblock(x)


class MelSpecVQVAE(nn.Module):
    """Generating mels from noise with VQVAE"""

    def __init__(self, hparams: MelSpecVQVAEHParams) -> None:
        super().__init__()
        self.hparams: MelSpecVQVAEHParams = hparams

        self.embedding_dim = hparams.embedding_dim
        self.num_embeddings = hparams.num_embeddings
        self.vq_weight = hparams.vq_weight

        self._build_encoder(
            hparams.conv_filters,
            hparams.conv_kernels,
            hparams.conv_strides,
            hparams.conv_padding
        )

        self.vq_layer = VectorQuantizer(
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
            beta=hparams.vq_beta
        )

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
        blocks: list[Any] = []
        in_channels = 1
        relu_fn = nn.LeakyReLU()

        for (
            c_filter,
            c_kernel,
            c_stride,
            c_padd
        ) in zip(
            conv_filters,
            conv_kernels,
            conv_strides,
            conv_padding
        ):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=c_filter,
                        kernel_size=c_kernel,
                        stride=c_stride,
                        padding=c_padd
                    ),
                    relu_fn
                )
            )
            in_channels = c_filter

        for _ in range(6):
            blocks.append(ResidualLayer(in_channels, in_channels))
        blocks.append(relu_fn)

        blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.embedding_dim,
                    kernel_size=1,
                    stride=1
                ),
                relu_fn
            )
        )

        self.encoder = nn.Sequential(*blocks)

    def _build_decoder(
        self,
        conv_filters: list,
        conv_kernels: list,
        conv_strides: list,
        deconv_padding: list,
        deconv_out_padding: list
    ):
        # Build Decoder
        blocks: list[Any] = []
        relu_fn = nn.LeakyReLU()

        for _ in range(6):
            blocks.append(ResidualLayer(conv_filters[-1], conv_filters[-1]))

        blocks.append(relu_fn)

        for i in reversed(range(1, len(conv_filters))):
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=conv_filters[i],
                        out_channels=conv_filters[i - 1],
                        kernel_size=conv_kernels[i],
                        stride=conv_strides[i],
                        padding=deconv_padding[i],
                        output_padding=deconv_out_padding[i]
                    ),
                    relu_fn
                )
            )

        blocks.append(
            nn.ConvTranspose2d(
                in_channels=conv_filters[0],
                out_channels=1,
                kernel_size=conv_kernels[0],
                stride=conv_strides[0],
                padding=deconv_padding[0],
                output_padding=deconv_out_padding[0]
            )
        )

        self.decoder = nn.Sequential(*blocks)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        encoded = self.encoder(x)
        return encoded

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        """

        result = self.decoder(z)
        return result

    def forward(self, x: torch.Tensor):
        encoded = self.encode(x)
        q_x, vq_loss = self.vq_layer(encoded)
        return [self.decode(q_x), x, vq_loss]

    def loss_function(
        self,
        y_recon: torch.Tensor,
        y_target: torch.Tensor,
        vq_loss,
    ) -> dict[str, Any]:
        recons_loss = F.mse_loss(y_recon, y_target)

        loss = recons_loss + self.vq_weight * vq_loss
        return {'loss': loss,
                'recon': recons_loss,
                'vq': vq_loss}

    def inference(self, x: torch.Tensor):
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x)[0]
