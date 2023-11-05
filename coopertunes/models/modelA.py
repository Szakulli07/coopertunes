from .model import Model
from ..hparams import modelAHParams


class ModelA(Model):
    """Example modelA"""

    def __init__(self, hparams: modelAHParams):
        super().__init__(hparams)

    def forward(self, **kwargs):
        """
        Returns data after forward.
        Calculate gradients.
        """

    def inference(self, **kwargs):
        """
        Returns data after forward.
        Does not calculate gradients.
        """
