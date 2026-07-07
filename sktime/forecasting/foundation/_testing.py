"""Testing helpers for foundation-model forecaster infrastructure."""

from dataclasses import dataclass

import numpy as np

from sktime.forecasting.foundation._base import BaseFoundationForecaster
from sktime.forecasting.foundation._config import (
    FineTuneConfig,
    FoundationModelSpec,
    InferenceConfig,
    ModelLoadConfig,
)
from sktime.forecasting.foundation._result import (
    ForecastRequest,
    ForecastResult,
    ModelContext,
    ModelHandle,
)


@dataclass
class FakeModel:
    """Tiny model object used by foundation infrastructure tests."""

    offset: float = 0.0
    trained: bool = False


class FakeFoundationForecaster(BaseFoundationForecaster):
    """Small foundation forecaster used for base-class conformance tests."""

    _foundation_spec = FoundationModelSpec(
        family="fake",
        default_model_path=None,
        dependency_group="none",
        supports_zero_shot=True,
        supports_pretrain=True,
        supports_fit_fine_tune=True,
        supports_peft=False,
        supports_quantization=False,
        supports_quantiles=True,
        supports_multivariate=False,
        supports_exogenous=True,
    )

    _tags = {
        "requires-fh-in-fit": False,
        "capability:pred_int": True,
        "capability:pretrain": True,
        "capability:exogenous": True,
        "capability:insample": False,
        "y_inner_mtype": "pd.Series",
        "X-y-must-have-same-index": False,
    }

    def __init__(
        self,
        offset=0.0,
        model_path=None,
        cache_suffix=None,
        fit_strategy="zero-shot",
    ):
        self.offset = offset
        self.model_path = model_path
        self.cache_suffix = cache_suffix
        self.fit_strategy = fit_strategy
        super().__init__()

    def _get_model_load_config(self) -> ModelLoadConfig:
        """Return fake model load config."""
        return ModelLoadConfig(
            model_path=self.model_path,
            extra_load_kwargs={"cache_suffix": self.cache_suffix},
        )

    def _get_fine_tune_config(self) -> FineTuneConfig:
        """Return fake tuning policy."""
        return FineTuneConfig(strategy=self.fit_strategy)

    def _get_inference_config(self) -> InferenceConfig:
        """Return fake inference policy."""
        return InferenceConfig()

    def _load_model(
        self,
        load: ModelLoadConfig,
        tune: FineTuneConfig,
    ) -> ModelHandle:
        """Load fake model state."""
        model = FakeModel(offset=self.offset)
        return ModelHandle(model=model)

    def _make_context(self, y, X, fh, cutoff, handle: ModelHandle) -> ModelContext:
        """Convert fitted data to fake model context."""
        return ModelContext(
            values=np.asarray(y),
            metadata={"cutoff": cutoff, "name": getattr(y, "name", None)},
        )

    def _predict_native(
        self,
        handle: ModelHandle,
        context: ModelContext,
        request: ForecastRequest,
    ) -> ForecastResult:
        """Return deterministic fake forecasts."""
        relative_fh = np.asarray(request.relative_fh, dtype=float)
        point = handle.model.offset + relative_fh
        quantiles = None
        if request.alpha is not None:
            quantiles = {alpha: point + alpha for alpha in request.alpha}
        return ForecastResult(mean=point, quantiles=quantiles)

    def _make_training_data(self, y, X, fh, purpose, handle: ModelHandle):
        """Return fake training data."""
        return {"y": y, "purpose": purpose}

    def _train_model(
        self,
        handle: ModelHandle,
        training_data,
        tune: FineTuneConfig,
    ) -> ModelHandle:
        """Return a mutable fake trained handle."""
        model = FakeModel(offset=handle.model.offset, trained=True)
        return ModelHandle(
            model=model,
            metadata={"artifact": "fake-artifact"},
            shareable=False,
            mutable=True,
        )
