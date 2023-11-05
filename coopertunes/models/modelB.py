from .model import Model
from ..hparams import modelBHParams


class ModelB(Model):
    """Example modelB"""

    def __init__(self, hparams: modelBHParams):
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
