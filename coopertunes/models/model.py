from abc import abstractmethod

from torch import nn

from ..hparams import HParams


class Model(nn.Module):
    """Abstract class for all coopertunes models"""

    @abstractmethod
    def __init__(self, hparams: HParams):
        super().__init__()
        self.hparams: HParams = hparams

    @abstractmethod
    def forward(self, **kwargs):
        """
        Returns data after forward.
        Calculate gradients.
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self, **kwargs):
        """
        Returns data after forward.
        Does not calculate gradients.
        """
        raise NotImplementedError
