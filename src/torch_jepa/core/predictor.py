from abc import ABC, abstractmethod
from typing import Generic

from torch_jepa._typing import TEncoding, TLatent
from torch_jepa.core.struct import DimMixin


class BasePredictor(Generic[TEncoding, TLatent], DimMixin, ABC):
    """Base class for a JEPA predictor."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)

    @abstractmethod
    def forward(self, encoding: TEncoding, latent: TLatent) -> TEncoding:
        """
        Predicts the target encoding from the context encoding
        and the conditioning latent variable.

        Args:
            context_encoding (TEncoding):
                The encoding of the context observation.
            latent (TLatent):
                The latent conditioning variable.

        Returns:
            TEncoding: The prediction for the encoding of the
                target observation.
        """
        pass
