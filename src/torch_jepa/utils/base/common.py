from typing import Generic, TypeVar

import torch
from tensordict import TensorClass

from torch_jepa._typing import TEncoding, TLatent, TObservation


class DimensionalModule(torch.nn.Module):
    """Base class for a module with some dimensionality."""

    def __init__(self, dim: int) -> None:
        super().__init__()

        self._dim: int = dim

    @property
    def dim(self) -> int:
        """Dimensionality of the module."""
        return self._dim


class BaseJEPALoss(TensorClass):
    """Base tensorclass for a JEPA loss."""

    total: torch.Tensor
    """The total loss calculated for the pass through the JEPA."""


TLoss = TypeVar("TLoss", bound=BaseJEPALoss)


class JEPAOutput(Generic[TObservation, TEncoding, TLatent, TLoss], TensorClass):
    """
    Output of a JEPA.
    Includes extra variables which might be required for downstream
    tasks or logging.
    """

    context: TObservation
    """The context observation passed into the JEPA."""

    target: TObservation
    """The target observation passed into the JEPA."""

    context_encoding: TEncoding
    """The output encoding of the context encoder."""

    latent: TLatent
    """The latent conditioning variable."""

    prediction: TEncoding
    """The prediction for the encoding of the target observation."""

    target_encoding: TEncoding
    """The output encoding from the target encoder."""

    loss: TLoss | None = None
    """The loss calculated for the pass through the JEPA."""
