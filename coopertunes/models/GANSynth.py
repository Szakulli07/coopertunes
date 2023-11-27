from torch import nn
import torch

from coopertunes.models import Model


EPS = 1e-8


class Generator(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(317, 256, (2, 16)),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.Upsample(size=(4, 32))
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.Upsample(size=(8, 64))
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.Upsample(size=(16, 128))
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.Upsample(size=(32, 256))
        )
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.Upsample(size=(64, 512))
        )
        self.block6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.Upsample(size=(128, 1024))
        )
        self.block7 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.ConvTranspose2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            PixelNormalization(EPS),
            nn.ConvTranspose2d(32, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, pitch):
        x = torch.concat((z, pitch), dim=1)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x


class PixelNormalization(nn.Module):

    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * (((x**2).mean(dim=1, keepdim=True) + self.eps).rsqrt())


class Discriminator(nn.Module):

    def __init__(self, hparams):
        pass

    def forward(self, noise):
        pass


if __name__ == "__main__":
    batch_size = 16
    generator = Generator(None)
    noise = torch.randn(batch_size, 256)
    pitch = torch.zeros(batch_size, 61)
    pitch[:, 3] = 1
    x = generator(noise, pitch)
    i = 0
