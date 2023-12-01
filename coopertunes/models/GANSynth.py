import torch
from torch import nn
from torch.nn import functional as F

from coopertunes.hparams.GANSynth import (DiscriminatorHParams,
                                          GANSynthHParams, GeneratorHParams)
from coopertunes.utils import PixelNormalization, dconv_same_padding


class Generator(nn.Module):

    def __init__(self, hparams: GeneratorHParams):
        super().__init__()
        self.activation_function = nn.LeakyReLU(hparams.leaky_relu_slope)

        block_idx = 0
        dconv_filters = hparams.block_dconv_filters[block_idx]
        dconv_kernel = hparams.block_dconv_kernel[block_idx]
        upsample = hparams.block_upsample_factor[block_idx]
        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(hparams.latent_dim + hparams.pitch_dim, dconv_filters, hparams.first_dconv_kernel),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.ConvTranspose2d(dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.Upsample(scale_factor=upsample)
        )

        block_idx += 1
        previous_dconv_filters = dconv_filters
        dconv_filters = hparams.block_dconv_filters[block_idx]
        dconv_kernel = hparams.block_dconv_kernel[block_idx]
        upsample = hparams.block_upsample_factor[block_idx]
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(previous_dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.ConvTranspose2d(dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.Upsample(scale_factor=upsample)
        )

        block_idx += 1
        previous_dconv_filters = dconv_filters
        dconv_filters = hparams.block_dconv_filters[block_idx]
        dconv_kernel = hparams.block_dconv_kernel[block_idx]
        upsample = hparams.block_upsample_factor[block_idx]
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(previous_dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.ConvTranspose2d(dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.Upsample(scale_factor=upsample)
        )

        block_idx += 1
        previous_dconv_filters = dconv_filters
        dconv_filters = hparams.block_dconv_filters[block_idx]
        dconv_kernel = hparams.block_dconv_kernel[block_idx]
        upsample = hparams.block_upsample_factor[block_idx]
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(previous_dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.ConvTranspose2d(dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.Upsample(scale_factor=upsample)
        )

        block_idx += 1
        previous_dconv_filters = dconv_filters
        dconv_filters = hparams.block_dconv_filters[block_idx]
        dconv_kernel = hparams.block_dconv_kernel[block_idx]
        upsample = hparams.block_upsample_factor[block_idx]
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(previous_dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.ConvTranspose2d(dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.Upsample(scale_factor=upsample)
        )

        block_idx += 1
        previous_dconv_filters = dconv_filters
        dconv_filters = hparams.block_dconv_filters[block_idx]
        dconv_kernel = hparams.block_dconv_kernel[block_idx]
        upsample = hparams.block_upsample_factor[block_idx]
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(previous_dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.ConvTranspose2d(dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.Upsample(scale_factor=upsample)
        )

        block_idx += 1
        previous_dconv_filters = dconv_filters
        dconv_filters = hparams.block_dconv_filters[block_idx]
        dconv_kernel = hparams.block_dconv_kernel[block_idx]
        self.block6 = nn.Sequential(
            nn.ConvTranspose2d(previous_dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.ConvTranspose2d(dconv_filters, dconv_filters, dconv_kernel, padding=dconv_same_padding(dconv_kernel)),
            self.activation_function,
            PixelNormalization(hparams.eps),
            nn.ConvTranspose2d(dconv_filters, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, pitch):
        x = torch.concat((z, pitch), dim=1)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, hparams: DiscriminatorHParams):
        super().__init__()
        self.activation_function = nn.LeakyReLU(hparams.leaky_relu_slope)

        block_idx = 0
        conv_filters = hparams.block_conv_filters[block_idx]
        conv_kernel = hparams.block_conv_kernel[block_idx]
        downsample = hparams.block_downsample_factor[block_idx]
        self.block0 = nn.Sequential(
            nn.Conv2d(2, conv_filters, 1, padding="same"),
            nn.Conv2d(conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.Conv2d(conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.AvgPool2d(downsample)
        )

        block_idx += 1
        previous_conv_filters = conv_filters
        conv_filters = hparams.block_conv_filters[block_idx]
        conv_kernel = hparams.block_conv_kernel[block_idx]
        downsample = hparams.block_downsample_factor[block_idx]
        self.block1 = nn.Sequential(
            nn.Conv2d(previous_conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.Conv2d(conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.AvgPool2d(downsample)
        )

        block_idx += 1
        previous_conv_filters = conv_filters
        conv_filters = hparams.block_conv_filters[block_idx]
        conv_kernel = hparams.block_conv_kernel[block_idx]
        downsample = hparams.block_downsample_factor[block_idx]
        self.block2 = nn.Sequential(
            nn.Conv2d(previous_conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.Conv2d(conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.AvgPool2d(downsample)
        )

        block_idx += 1
        previous_conv_filters = conv_filters
        conv_filters = hparams.block_conv_filters[block_idx]
        conv_kernel = hparams.block_conv_kernel[block_idx]
        downsample = hparams.block_downsample_factor[block_idx]
        self.block3 = nn.Sequential(
            nn.Conv2d(previous_conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.Conv2d(conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.AvgPool2d(downsample)
        )

        block_idx += 1
        previous_conv_filters = conv_filters
        conv_filters = hparams.block_conv_filters[block_idx]
        conv_kernel = hparams.block_conv_kernel[block_idx]
        downsample = hparams.block_downsample_factor[block_idx]
        self.block4 = nn.Sequential(
            nn.Conv2d(previous_conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.Conv2d(conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.AvgPool2d(downsample)
        )

        block_idx += 1
        previous_conv_filters = conv_filters
        conv_filters = hparams.block_conv_filters[block_idx]
        conv_kernel = hparams.block_conv_kernel[block_idx]
        downsample = hparams.block_downsample_factor[block_idx]
        self.block5 = nn.Sequential(
            nn.Conv2d(previous_conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.Conv2d(conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.AvgPool2d(downsample)
        )

        block_idx += 1
        previous_conv_filters = conv_filters
        conv_filters = hparams.block_conv_filters[block_idx]
        conv_kernel = hparams.block_conv_kernel[block_idx]
        self.block6 = nn.Sequential(
            nn.Conv2d(previous_conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
            nn.Conv2d(conv_filters, conv_filters, conv_kernel, padding="same"),
            self.activation_function,
        )
        self.pitch_classifier = nn.Linear(hparams.linear_in_size, hparams.pitch_dim)
        self.discriminator_output = nn.Linear(hparams.linear_in_size, 1)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = torch.flatten(x, 1)
        dis_out = self.discriminator_output(x)
        pitch = F.softmax(self.pitch_classifier(x), dim=0)
        return (dis_out, pitch)


if __name__ == "__main__":
    batch_size = 16
    hparams = GANSynthHParams()
    generator = Generator(hparams.generator)
    discriminator = Discriminator(hparams.discriminator)
    noise = torch.randn(batch_size, 256)
    pitch = torch.zeros(batch_size, 61)
    pitch[:, 3] = 1
    x = generator(noise, pitch)
    y = discriminator(x)
    i = 0
