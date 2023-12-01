from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam

from coopertunes.hparams.GANSynth import GANSynthHParams


class GANSynthSupervisor:

    def __init__(self, models: Tuple, device, hparams: GANSynthHParams):
        self.generator = models[0]
        self.discriminator = models[1]
        self.device = device
        self.hparams = hparams

        self.generator_optimizer = Adam(self.generator.params(), lr=hparams.generator.lr, betas=hparams.generator.betas)
        self.discriminator_optimizer = Adam(self.discriminator.params(), lr=hparams.discriminator.lr, betas=hparams.discriminator.betas)

        self.train_loader = None

    def train(self):
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator.train()
        self.discriminator.train()

        criterion = nn.MSELoss()

        # TODO: Pitch
        for i in range(self.hparams.epochs):
            for data in self.train_loader:
                self.discriminator_optimizer.zero_grad()
                real_images = data[0].to(self.device)
                batch_size = real_images.size(0)
                label = torch.ones((batch_size,), dtype=torch.float, device=self.device)
                output = self.discriminator(real_images).view(-1)
                error_discriminator_real = criterion(output, label)
                error_discriminator_real.backward()

                noise = torch.randn(batch_size, self.hparams.generator.latent_dim, device=self.device)
                fake_images = self.generator(noise, None)                                               # TODO: Add pitch
                label_fake = torch.zeros((batch_size,), dtype=torch.float, device=self.device)
                output = self.discriminator(fake_images.detach()).view(-1)
                error_discriminator_fake = criterion(output, label_fake)
                error_discriminator_fake.backward()
                error_discriminator = error_discriminator_real + error_discriminator_fake
                self.discriminator_optimizer.step()

                self.generator_optimizer.zero_grad()
                label = torch.ones((batch_size,), dtype=torch.float, device=self.device)
                output = self.discriminator(fake_images).view(-1)
                error_generator = criterion(output, label)
                error_generator.backward()
                self.generator_optimizer.step()




