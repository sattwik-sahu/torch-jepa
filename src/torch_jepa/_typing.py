from typing import TypeVar, Union

import torch
from tensordict import TensorClass, TensorDict

TensorData = Union[torch.Tensor, TensorDict, TensorClass]
"""Unified data type for tensors, tensordicts, and tensorclasses."""

TObservation = TypeVar("TObservation", bound=TensorData)
"""Generic data type for vars in input observation space."""

TEncoding = TypeVar("TEncoding", bound=TensorData)
"""Generic data type for encoder output encoding."""

TLatent = TypeVar("TLatent", bound=TensorData)
"""Generic data type for latent conditioning, for predictor."""
