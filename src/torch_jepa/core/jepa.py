from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from typing_extensions import override

from torch_jepa._typing import TEncoding, TLatent, TObservation
from torch_jepa.core.encoder import BaseEncoder
from torch_jepa.core.predictor import BasePredictor
from torch_jepa.core.struct import BaseJEPALoss, DimMixin, JEPAOutput

TLoss = TypeVar("TLoss", bound=BaseJEPALoss)


class BaseJointEmbeddingPredictiveArchitecture(
    Generic[TObservation, TEncoding, TLatent, TLoss],
    DimMixin,
    ABC,
):
    """Base class for JEPA implementation."""

    def __init__(
        self,
        context_encoder: BaseEncoder[TObservation, TEncoding],
        predictor: BasePredictor[TEncoding, TLatent],
        target_encoder: BaseEncoder[TObservation, TEncoding] | None = None,
    ) -> None:
        super().__init__(dim=context_encoder.dim)

        # Check if context encoder and predictor are compatible
        assert context_encoder.dim == predictor.dim, (
            f"Got encoder, predictor of different dimensionalities ({context_encoder.dim} and {predictor.dim})"
        )

        self._context_encoder: BaseEncoder[TObservation, TEncoding] = context_encoder
        self._target_encoder: BaseEncoder[TObservation, TEncoding] = (
            target_encoder if target_encoder is not None else context_encoder
        )
        self._predictor: BasePredictor[TEncoding, TLatent] = predictor

    @property
    def context_encoder(self) -> BaseEncoder[TObservation, TEncoding]:
        return self._context_encoder

    @property
    def target_encoder(self) -> BaseEncoder[TObservation, TEncoding]:
        return self._target_encoder

    @property
    def predictor(self) -> BasePredictor[TEncoding, TLatent]:
        return self._predictor

    @abstractmethod
    def _calculate_loss(
        self,
        context: TObservation,
        target: TObservation,
        context_encoding: TEncoding,
        latent: TLatent,
        prediction: TEncoding,
        target_encoding: TEncoding,
    ) -> TLoss:
        pass

    @abstractmethod
    def _update_target_encoder_weights(self) -> None:
        pass

    @override
    def forward(
        self,
        context: TObservation,
        target: TObservation,
        latent: TLatent,
        calculate_loss: bool = True,
    ) -> JEPAOutput[TObservation, TEncoding, TLatent, TLoss]:
        context_encoding: TEncoding = self._context_encoder(observation=context)
        target_encoding: TEncoding = self._target_encoder(observation=target)
        prediction: TEncoding = self._predictor(
            encoding=context_encoding, latent=latent
        )

        if calculate_loss:
            loss: TLoss = self._calculate_loss(
                context=context,
                target=target,
                context_encoding=context_encoding,
                latent=latent,
                prediction=prediction,
                target_encoding=target_encoding,
            )
        else:
            loss = None

        return JEPAOutput(
            context=context,
            target=target,
            context_encoding=context_encoding,
            latent=latent,
            prediction=prediction,
            target_encoding=target_encoding,
            loss=loss,
        )
