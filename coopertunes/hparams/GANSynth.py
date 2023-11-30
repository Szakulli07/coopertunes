from coopertunes.hparams.hparams import HParams


class GANSynthHParams(HParams):

    def __init__(self):
        super().__init__()
        self.generator = GeneratorHParams()
        self.discriminator = DiscriminatorHParams()

        assert self.generator.pitch_dim == self.discriminator.pitch_dim


class GeneratorHParams:

    def __init__(self):
        self.latent_dim = 256
        self.pitch_dim = 61

        self.leaky_relu_slope = 0.2
        self.block_dconv_filters = [256, 256, 256, 256, 128, 64, 32]
        self.block_upsample_factor = [2] * 6
        self.first_dconv_kernel = (2, 16)
        self.block_dconv_kernel = [3] * 7

        self.eps = 1e-8



class DiscriminatorHParams:
    
    def __init__(self):
        self.leaky_relu_slope = 0.2
        self.block_conv_filters = [32, 64, 128, 256, 256, 256, 256]
        self.block_conv_kernel = [3] * 7
        self.block_downsample_factor = [2] * 6

        self.linear_in_size = 256 * 2 * 16
        self.pitch_dim = 61
