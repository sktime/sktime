"""Shared base class for foundation-model forecasters."""

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.foundation._cache import FOUNDATION_MODEL_CACHE, _stable_repr
from sktime.forecasting.foundation._result import ModelHandle
from sktime.utils.warnings import warn


class BaseFoundationForecaster(BaseForecaster):
    """Shared base class for pretrained/foundation forecasting models."""

    def __init__(
        self,
        model_path=None,
        tokenizer_path=None,
        config=None,
        load_kwargs=None,
        quantization_config=None,
        device=None,
        forward_kwargs=None,
        random_state=None,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config = config
        self.load_kwargs = load_kwargs
        self.quantization_config = quantization_config
        self.device = device
        self.forward_kwargs = forward_kwargs
        self.random_state = random_state

        super().__init__()

    def __post_init__(self):
        """Post-init constructor logic for shared foundation-model parameters."""
        self.random_state_ = check_random_state(self.random_state)

    def _fit(self, y, X=None, fh=None):
        """Fit zero-shot foundation forecaster context and load backend state."""
        if X is None:
            if self.get_tag("capability:exogenous", False):
                warn(
                    f"{self.__class__.__name__} received no X. Placeholder "
                    "covariates will be created from y.",
                    obj=self,
                )
            X = self._make_fallback_X(y)

        self.y_name_ = getattr(y, "name", None)
        self.y_context_ = y.copy()
        self.X_context_ = None if X is None else X.copy()
        self.max_context_ = len(self.y_context_)

        self._prepare_foundation_context(y=y, X=X, fh=fh)
        self.model_handle_ = self._load_foundation_backend()
        self._fit_foundation_context(y=y, X=X, fh=fh, handle=self.model_handle_)
        return self

    def _predict(self, fh, X=None):
        """Forecast point predictions from foundation-model sample paths."""
        prediction_result, y_index = self._predict_samples(fh=fh, X=X)
        return self._format_point_predictions(
            prediction_result=prediction_result,
            y_index=y_index,
            fh=fh,
        )

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Forecast quantiles from foundation-model sample paths."""
        prediction_result, y_index = self._predict_samples(fh=fh, X=X)
        return self._format_quantile_predictions(
            prediction_result=prediction_result,
            y_index=y_index,
            fh=fh,
            alpha=alpha,
        )

    def _format_point_predictions(self, prediction_result, y_index, fh):
        """Format point predictions from sample paths."""
        point_pred = prediction_result.median(axis=1).to_numpy()
        return pd.Series(point_pred, index=y_index, name=self.y_name_)

    def _format_quantile_predictions(self, prediction_result, y_index, fh, alpha):
        """Format quantile predictions from sample paths."""
        quantiles = np.quantile(prediction_result.to_numpy(), q=alpha, axis=1).T
        columns = pd.MultiIndex.from_product([self._get_varnames(), alpha])
        return pd.DataFrame(quantiles, index=y_index, columns=columns)

    def _predict_samples(self, fh, X=None):
        """Generate native sample paths after applying shared prediction setup."""
        self._seed_torch_if_requested()
        return self._predict_samples_native(fh=fh, X=X)

    def _seed_torch_if_requested(self):
        """Seed torch for deterministic native sampling when random_state is set."""
        if self.random_state is None:
            return

        import torch

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _load_foundation_backend(self):
        """Load or retrieve cached backend state as a foundation model handle."""
        handle = getattr(self, "model_handle_", None)
        if handle is not None:
            self._unpack_model_handle(handle)
            return handle

        handle = FOUNDATION_MODEL_CACHE.get_or_load(
            key=self._get_unique_key(),
            loader=self._load_model_handle,
        )
        self._unpack_model_handle(handle)
        return handle

    def _load_model_handle(self):
        """Load native backend objects and wrap them in a model handle."""
        tokenizer = self._load_native_tokenizer(self.tokenizer_path)
        model = self._load_native_model(self.model_path)
        handle = ModelHandle(model=model, tokenizer=tokenizer)
        return self._prepare_model_handle(handle)

    def _prepare_model_handle(self, handle):
        """Apply common post-load setup to a model handle."""
        handle.tokenizer = self._to_device(handle.tokenizer)
        handle.model = self._to_device(handle.model)
        handle.pipeline = self._to_device(handle.pipeline)
        return handle

    def _unpack_model_handle(self, handle):
        """Expose model handle members under legacy fitted attribute names."""
        self.tokenizer_ = handle.tokenizer
        self.model_ = handle.model
        self.pipeline_ = handle.pipeline

    def _to_device(self, obj):
        """Move backend object to device and eval mode if supported."""
        if obj is None:
            return None

        device = self._get_backend_device()
        if device is not None and hasattr(obj, "to"):
            obj = obj.to(device)
        if hasattr(obj, "eval"):
            obj = obj.eval()
        return obj

    def _get_unique_key(self):
        """Build cache key for foundation backend loading."""
        return (
            self.__class__.__name__,
            self.model_path,
            self.tokenizer_path,
            self._get_backend_device(),
            _stable_repr(self.config),
            _stable_repr(self.load_kwargs),
            _stable_repr(self.quantization_config),
            _stable_repr(self._cache_key_extra()),
        )

    def __getstate__(self):
        """Return pickle state without loaded backend objects."""
        state = self.__dict__.copy()
        state["model_handle_"] = None
        state["tokenizer_"] = None
        state["model_"] = None
        state["pipeline_"] = None
        return state

    def __setstate__(self, state):
        """Restore estimator state from pickle."""
        self.__dict__.update(state)

    def _prepare_foundation_context(self, y, X, fh):
        """Prepare estimator-specific context needed before backend loading."""

    def _fit_foundation_context(self, y, X, fh, handle):
        """Customize estimator-specific fitted context after backend loading."""

    def _make_fallback_X(self, y):
        """Create placeholder covariates from y when X is omitted."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support automatic X fallback"
        )

    def _predict_samples_native(self, fh, X=None):
        """Generate model-native sample paths."""
        raise NotImplementedError

    def _load_native_tokenizer(self, tokenizer_path):
        """Load a native tokenizer object, if the backend uses one."""
        return None

    def _load_native_model(self, model_path):
        """Load a native model object."""
        raise NotImplementedError

    def _cache_key_extra(self):
        """Return estimator-specific cache-key components."""
        return ()

    def _get_backend_device(self):
        """Return device used for cached backend objects."""
        return self.device
