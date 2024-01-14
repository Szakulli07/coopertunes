from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam

from coopertunes.hparams.GANSynth import GANSynthHParams
from coopertunes.models.GANSynth import Discriminator, Generator


class GANSynthSupervisor:

    def __init__(self, models: Tuple, device, hparams: GANSynthHParams):
        self.generator = models[0]
        self.discriminator = models[1]
        self.device = device
        self.hparams = hparams

        self.generator_optimizer = Adam(self.generator.parameters(), lr=hparams.generator.lr, betas=hparams.generator.betas)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=hparams.discriminator.lr, betas=hparams.discriminator.betas)

        # TODO: real data
        self.train_loader = None

    def train(self):
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator.train()
        self.discriminator.train()

        source_criterion = nn.BCEWithLogitsLoss()
        pitch_criterion = nn.CrossEntropyLoss()

        for epoch in range(self.hparams.epochs):
            for data in self.train_loader:
                self.discriminator_optimizer.zero_grad()
                real_images = data[0].to(self.device)
                real_pitch = data[1].to(self.device)
                batch_size = real_images.size(0)
                label = torch.ones((batch_size,), dtype=torch.float, device=self.device)
                (discriminator_output, discriminator_pitch) = self.discriminator(real_images)                       # .view(-1)
                error_discriminator_real = source_criterion(discriminator_output.view(-1), label) + pitch_criterion(discriminator_pitch, real_pitch)
                error_discriminator_real.backward()

                noise = torch.randn(batch_size, self.hparams.generator.latent_dim, device=self.device)
                # TODO: change to random pitch
                fake_pitch = torch.zeros(batch_size, 61)
                fake_pitch[:, 3] = 1
                fake_images = self.generator(noise, fake_pitch)
                label_fake = torch.zeros((batch_size,), dtype=torch.float, device=self.device)
                (discriminator_output, discriminator_pitch) = self.discriminator(fake_images.detach())
                error_discriminator_fake = source_criterion(discriminator_output.view(-1), label_fake) + pitch_criterion(discriminator_pitch, fake_pitch)
                error_discriminator_fake.backward()
                error_discriminator = error_discriminator_real + error_discriminator_fake
                self.discriminator_optimizer.step()

                self.generator_optimizer.zero_grad()
                label = torch.ones((batch_size,), dtype=torch.float, device=self.device)
                (discriminator_output, discriminator_pitch) = self.discriminator(fake_images)
                error_generator = source_criterion(discriminator_output.view(-1), label) + pitch_criterion(discriminator_pitch, fake_pitch)
                error_generator.backward()
                self.generator_optimizer.step()


if __name__ == "__main__":
    hparams = GANSynthHParams()
    generator = Generator(hparams.generator)
    discriminator = Discriminator(hparams.discriminator)
    supervisor = GANSynthSupervisor((generator, discriminator), torch.device("cpu"), hparams)
    supervisor.train()
