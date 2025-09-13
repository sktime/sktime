# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from __future__ import annotations
import logging
from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from ..api_adapter.forecast import ForecastModel

LOGGER = logging.getLogger()


class TensorQuantileUniPredictMixin(ForecastModel):
    @abstractmethod
    def _forecast_tensor(
        self,
        context: Any,
        prediction_length: int | None = None,
        **predict_kwargs,
    ) -> Any:
        pass

    @property
    @abstractmethod
    def quantiles(self):
        pass

    def _forecast_quantiles(
        self,
        context: Any,
        prediction_length: int | None = None,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        output_device: str = "cpu",
        auto_cast: bool = False,
        **predict_kwargs,
    ) -> tuple[Any, Any]:
        import torch

        with torch.autocast(device_type=self.device.type, enabled=auto_cast):
            predictions = self._forecast_tensor(
                context=context, prediction_length=prediction_length, **predict_kwargs
            ).detach()
        predictions = predictions.to(torch.device(output_device)).swapaxes(1, 2)

        training_quantile_levels = list(self.quantiles)

        if set(quantile_levels).issubset(set(training_quantile_levels)):
            quantiles = predictions[
                ..., [training_quantile_levels.index(q) for q in quantile_levels]
            ]
        else:
            if min(quantile_levels) < min(training_quantile_levels) or max(
                quantile_levels
            ) > max(training_quantile_levels):
                logging.warning(
                    f"Requested quantile levels ({quantile_levels}) fall outside the range of "
                    f"quantiles the model was trained on ({training_quantile_levels}). "
                    "Predictions for out-of-range quantiles will be clamped to the nearest "
                    "boundary of the trained quantiles (i.e., minimum or maximum trained level). "
                    "This can significantly impact prediction accuracy, especially for extreme quantiles. "
                )
            # Interpolate quantiles
            augmented_predictions = torch.cat(
                [predictions[..., [0]], predictions, predictions[..., [-1]]],
                dim=-1,
            )
            quantiles = torch.quantile(
                augmented_predictions,
                q=torch.tensor(quantile_levels, dtype=augmented_predictions.dtype),
                dim=-1,
            ).permute(1, 2, 0)
        # median as mean
        mean = predictions[:, :, training_quantile_levels.index(0.5)]
        return quantiles, mean
