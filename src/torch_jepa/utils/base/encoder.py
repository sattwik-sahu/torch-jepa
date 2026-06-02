from abc import ABC, abstractmethod
from typing import Generic

from torch_jepa._typing import TEncoding, TObservation
from torch_jepa.utils.base.common import DimensionalModule


class BaseEncoder(Generic[TObservation, TEncoding], DimensionalModule, ABC):
    """Base class for a JEPA encoder."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim=dim)

    @abstractmethod
    def forward(self, observation: TObservation) -> TEncoding:
        """
        Encodes the observation into an encoding.

        Args:
            observation (TObservation): The observation.

        Returns:
            TEncoding: The encoding of the observation.
        """
        pass
