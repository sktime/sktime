"""Shared lifecycle and extension points for foundation-model forecasters.

This module contains the orchestration that is common to zero-shot forecasting
adapters. Model-specific modules are responsible only for loading their native
backend and translating between pandas data and :class:`ForecastResult`.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import numpy as np
from sklearn.utils import check_random_state

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.foundation._cache import FOUNDATION_MODEL_CACHE
from sktime.forecasting.foundation._format import (
    format_point_result,
    format_quantile_result,
)
from sktime.forecasting.foundation._result import ForecastRequest, ForecastResult
from sktime.forecasting.foundation._spec import FoundationModelSpec


class BaseFoundationForecaster(BaseForecaster):
    """Base class for zero-shot foundation-model forecasting adapters.

    Subclasses must implement two hooks:

    * :meth:`_load_model` loads the native model and returns a
      :class:`ModelHandle`.
    * :meth:`_inference` converts the stored pandas context to backend input and
      returns a :class:`ForecastResult`.

    The base class owns the rest of the forecasting lifecycle: storing context,
    caching loaded models, normalizing forecasting horizons, setting up Torch
    inference, and formatting point and quantile output according to the sktime
    API.

    Parameters
    ----------
    model_spec : FoundationModelSpec
        Model identity, loading options, and prediction options. A concrete
        estimator should expose meaningful public constructor parameters, assign
        them to attributes, build one ``FoundationModelSpec``, and pass it to
        ``super().__init__``. Standard fields such as ``device`` belong directly
        on the specification; backend-only options belong in
        ``load_extra_kwargs`` or ``predict_extra_kwargs``.

    Attributes
    ----------
    model_spec : FoundationModelSpec
        Active runtime specification. The specification passed by the constructor
        is replaced during ``__post_init__`` with a normalized copy. For example,
        ``device="auto"`` and string Torch dtypes are resolved here. Later
        fit-derived settings are also applied by replacing this attribute.
    context_y_ : pd.DataFrame
        Copy of the target passed to ``fit``, with shape
        ``(n_context_timepoints, n_target_variables)``.
    context_X_ : pd.DataFrame or None
        Copy of the exogenous data passed to ``fit``.
    model_handle_ : ModelHandle or None
        Native backend objects used for inference. Handles may be shared between
        estimator instances and are deliberately removed during serialization.

    Notes
    -----
    A typical subclass follows this workflow:

    1. Define estimator tags, including soft dependencies and capabilities.
    2. In ``__init__``, store every public parameter, construct a
       ``FoundationModelSpec``, and call ``super().__init__(model_spec=...)``.
    3. Implement ``_load_model`` and ``_inference``.
    4. Optionally implement ``_update_attrs_in_fit`` when shapes or loading
       settings depend on the fitted data. Use ``_update_model_spec`` for derived
       runtime settings that must participate in the model cache key.
    5. Override ``_get_unique_model_key`` if the default key does not contain all
       inputs that affect loaded model state.

    A minimal adapter has the following structure (``NativeModel`` represents the
    third-party backend)::

        class MyFoundationForecaster(BaseFoundationForecaster):
            _tags = {
                "python_dependencies": ["native-package"],
                "capability:pred_int": False,
            }

            def __init__(self, model_path="provider/checkpoint", device="auto"):
                self.model_path = model_path
                self.device = device
                model_spec = FoundationModelSpec(
                    model_path=model_path,
                    device=device,
                )
                super().__init__(model_spec=model_spec)

            def _load_model(self):
                model = NativeModel.from_pretrained(
                    self.model_spec.model_path
                ).to(self.model_spec.device)
                return ModelHandle(model=model)

            def _inference(
                self,
                handle,
                context_y,
                context_X,
                future_X,
                pred_len,
                fh,
                alpha=None,
            ):
                values = handle.model.predict(
                    context_y.to_numpy(),
                    prediction_length=pred_len,
                    **self.model_spec.predict_extra_kwargs,
                )
                return ForecastResult(mean=np.asarray(values))

    Set ``_uses_torch_inference_context = False`` for non-Torch backends. Consider
    the ``capability:multivariate``, ``capability:pred_int``, exogenous-data, and
    in-sample prediction tags when declaring adapter support. Override ``_predict``
    or ``_predict_quantiles`` only when the shared column/index formatting cannot
    represent a model's native output contract.

    ``_inference`` receives pandas ``DataFrame`` context and should return numeric
    arrays with time on axis 0 and target variables on axis 1. It may return a
    complete future horizon of shape ``(pred_len, n_targets)``; the base class then
    selects sparse requested steps. Alternatively, it may return exactly
    ``len(fh)`` rows in requested horizon order, which is useful for backends that
    natively support sparse, in-sample, or mixed horizons. Univariate output may
    be one-dimensional.

    Loaded handles are process-local shared state. Treat the model, tokenizer,
    and pipeline in a handle as read-only during prediction. Per-fit or per-series
    mutable state belongs on the estimator, not on the shared handle.
    """

    _tags = {
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "tests:vm": True,
    }
    _uses_torch_inference_context = True

    def __init__(self, model_spec: FoundationModelSpec):
        """Initialize the forecaster from a model runtime specification."""
        if not isinstance(model_spec, FoundationModelSpec):
            raise TypeError("model_spec must be a FoundationModelSpec")
        self.model_spec = model_spec
        super().__init__()

    def __dynamic_tags__(self):
        """Clear soft-dependency tags when dependency checks are disabled.

        Subclasses overriding this hook should call ``super().__dynamic_tags__()``
        before applying their model-specific dynamic tags.
        """
        if self.model_spec.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __post_init__(self):
        """Create the runtime specification after estimator initialization.

        ``BaseObject`` calls this hook automatically. Subclasses that override it
        must call ``super().__post_init__()`` before using normalized values from
        ``model_spec``.
        """
        model_spec = self.model_spec
        self.model_spec = replace(
            model_spec,
            config=self._resolve_config(model_spec.config),
            device=self._resolve_device(model_spec.device),
            dtype=self._resolve_dtype(model_spec.dtype),
            random_state=self._resolve_random_state(model_spec.random_state),
            load_extra_kwargs=model_spec.load_extra_kwargs,
            predict_extra_kwargs=model_spec.predict_extra_kwargs,
        )

    def _fit(self, y, X=None, fh=None):
        """Store zero-shot context and obtain the shared model handle.

        Parameters
        ----------
        y : pd.DataFrame
            Target context with shape ``(n_context_timepoints, n_targets)``.
        X : pd.DataFrame or None
            Past exogenous context in the estimator's internal mtype.
        fh : ForecastingHorizon or None
            Validated forecasting horizon supplied by ``BaseForecaster``.

        Returns
        -------
        self
            Fitted forecaster.

        Notes
        -----
        ``_update_attrs_in_fit`` runs before the model cache is queried so a
        subclass can derive loading settings from ``y``, ``X``, or ``fh``.
        """
        self._update_attrs_in_fit(y=y, X=X, fh=fh)
        self.context_y_ = y.copy()
        self.context_X_ = None if X is None else X.copy()
        self.model_handle_ = self._get_or_load_model_handle()
        return self

    def _predict(self, fh, X=None):
        """Forecast and format point predictions.

        Point output is chosen from ``ForecastResult`` in this order: mean,
        median, then the 0.5 quantile.
        """
        result, request = self._foundation_predict(fh=fh, X=X)
        return format_point_result(result=result, request=request, y=self.context_y_)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Forecast and format requested quantiles.

        Concrete classes must set the ``capability:pred_int`` tag to ``True`` to
        expose this shared implementation.
        """
        result, request = self._foundation_predict(fh=fh, X=X, alpha=alpha)
        return format_quantile_result(
            result=result,
            request=request,
            y=self.context_y_,
            alpha=alpha,
        )

    def _foundation_predict(self, fh, X=None, alpha=None):
        """Run model-specific inference with shared horizon and reload setup.

        ``alpha`` is normalized to a tuple before it reaches ``_inference``. The
        model handle is restored lazily when an estimator was deserialized.
        """
        if fh is None:
            fh = self.fh

        request = self._make_forecast_request(fh=fh, alpha=alpha)
        # Most backends generate a dense out-of-sample horizon. Adapters that
        # support in-sample or mixed horizons can inspect ``fh`` directly and
        # return one row per requested step instead.
        pred_len = max(request.relative_fh)

        self.model_handle_ = self._get_or_load_model_handle()
        with self._inference_context(handle=self.model_handle_):
            result = self._inference(
                handle=self.model_handle_,
                context_y=self.context_y_,
                context_X=self.context_X_,
                future_X=X,
                pred_len=pred_len,
                fh=fh,
                alpha=request.alpha,
            )
        if not isinstance(result, ForecastResult):
            raise TypeError(
                f"{self.__class__.__name__}._inference must return ForecastResult, "
                f"but returned {type(result).__name__}."
            )
        return result, request

    def _make_forecast_request(self, fh, alpha=None):
        """Normalize an sktime forecasting horizon for formatting.

        Parameters
        ----------
        fh : ForecastingHorizon
            Horizon already validated by ``BaseForecaster``.
        alpha : sequence of float or None
            Requested quantile probabilities.

        Returns
        -------
        ForecastRequest
            Relative integer steps, their absolute output index, and immutable
            quantile probabilities.
        """
        relative_fh = np.asarray(fh.to_relative(self.cutoff).to_pandas(), dtype=int)
        absolute_index = fh.to_absolute(self.cutoff).to_pandas()
        return ForecastRequest(
            relative_fh=tuple(int(value) for value in relative_fh),
            absolute_index=absolute_index,
            alpha=None if alpha is None else tuple(alpha),
        )

    def _get_or_load_model_handle(self):
        """Return an attached or process-cached model handle.

        The loader is called only on a cache miss. Because a cached handle can be
        returned to several estimator instances, adapters must not store
        series-specific mutable state in it.
        """
        handle = getattr(self, "model_handle_", None)
        if handle is not None:
            return handle

        return FOUNDATION_MODEL_CACHE.get_or_load(
            key=self._get_unique_model_key(),
            loader=self._load_model,
        )

    @contextmanager
    def _inference_context(self, handle):
        """Apply local Torch seeding, evaluation mode, and inference mode.

        The Torch random-number-generator state is restored on exit, so prediction
        does not alter application-level randomness. ``random_state`` seeds the
        local context. Non-Torch adapters must set
        ``_uses_torch_inference_context = False`` on the class.

        Parameters
        ----------
        handle : ModelHandle
            Handle whose ``model`` may expose ``eval`` and ``device``.
        """
        if not self._uses_torch_inference_context:
            yield
            return

        import torch

        model = handle.model
        if hasattr(model, "eval"):
            model.eval()

        devices = []
        model_device = getattr(model, "device", None)
        if model_device is not None:
            model_device = torch.device(model_device)
            if model_device.type == "cuda":
                devices = [model_device.index or torch.cuda.current_device()]

        with torch.random.fork_rng(devices=devices):
            if self.model_spec.random_state is not None:
                torch.manual_seed(self.model_spec.random_state)
            with torch.inference_mode():
                yield

    def _resolve_random_state(self, random_state):
        """Convert sklearn-compatible random state input to one integer seed."""
        rng = check_random_state(random_state)
        return (
            None if random_state is None else int(rng.randint(np.iinfo(np.int32).max))
        )

    def _resolve_config(self, config):
        """Return an isolated deep copy of model configuration.

        ``None`` is normalized to an empty dictionary so loading hooks can safely
        treat a missing config as a mapping. Deep copying supports both ordinary
        dictionaries and ``transformers.PretrainedConfig``-style objects without
        requiring child adapters to override this method.
        """
        if config is None:
            return {}

        return deepcopy(config)

    def _resolve_dtype(self, dtype):
        """Resolve ``"torch.<name>"`` strings to native Torch dtype objects.

        Native dtype objects and backend-specific values such as ``"auto"`` are
        returned unchanged. A string in the ``torch.`` namespace must identify an
        actual ``torch.dtype`` rather than another Torch attribute.
        """
        if not isinstance(dtype, str) or not dtype.startswith("torch."):
            return dtype

        import torch

        dtype_name = dtype.removeprefix("torch.")
        resolved_dtype = getattr(torch, dtype_name, None)
        if not isinstance(resolved_dtype, torch.dtype):
            raise ValueError(
                f"Unknown Torch dtype {dtype!r}. Expected a valid "
                "'torch.<dtype_name>' string."
            )
        return resolved_dtype

    def _resolve_device(self, device):
        """Resolve explicit, configured, or automatic Torch device selection.

        Non-Torch adapters should pass an explicit backend/device value or
        override this method if they support their own ``"auto"`` policy.
        """
        if device != "auto":
            return device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_unique_model_key(self):
        """Build a deterministic cache key from model-loading parameters.

        Prediction-only options, random seeds, and dependency-check settings do
        not change loaded model state and are intentionally omitted. ``config`` is
        included because it represents load-affecting model configuration. Omit
        it when an adapter's public ``config`` controls prediction only; the
        runtime value will then be an empty dictionary. Nested containers in the
        key are normalized by the cache.
        """
        model_spec = self.model_spec
        key_items = {
            "class": self.__class__.__name__,
            "config": model_spec.config,
            "device": model_spec.device,
            "dtype": model_spec.dtype,
            "model_path": model_spec.model_path,
            "quantization_config": model_spec.quantization_config,
            "revision": model_spec.revision,
            "tokenizer_path": model_spec.tokenizer_path,
            "load_extra_kwargs": model_spec.load_extra_kwargs,
        }
        return tuple(sorted(key_items.items()))

    @classmethod
    def _has_implementation_of(cls, method):
        """Respect probabilistic capability tags for shared quantile logic."""
        if method == "_predict_quantiles" and not cls.get_class_tag(
            "capability:pred_int", False
        ):
            return False
        return super()._has_implementation_of(method)

    def __getstate__(self):
        """Return pickle state without potentially unpickleable backend objects.

        The stored context and runtime specification remain serialized. A later
        prediction reloads or reuses a cached handle transparently.
        """
        state = self.__dict__.copy()
        state["model_handle_"] = None
        return state

    def _update_attrs_in_fit(self, y, X, fh):
        """Optionally derive model-specific fitted attributes before loading.

        Parameters
        ----------
        y : pd.DataFrame
            Target context, shape ``(n_context_timepoints, n_targets)``.
        X : pd.DataFrame or None
            Past exogenous context.
        fh : ForecastingHorizon or None
            Validated horizon available during fit.

        Notes
        -----
        The default implementation is a no-op. Use ``_update_model_spec`` if a
        derived value affects loading and therefore must be part of the cache key.
        Ordinary per-series state can be stored as a fitted attribute ending in
        ``_``.
        """

    def _load_model(self):
        """Load native backend state and return a :class:`ModelHandle`.

        This required hook is called without arguments on a process-local cache
        miss. Read normalized loading values and ``load_extra_kwargs`` from
        ``self.model_spec``. Put a primary neural network or equivalent object in
        ``handle.model``; optional tokenizers and high-level prediction pipelines
        have dedicated fields.

        The returned objects must depend only on settings represented by
        ``_get_unique_model_key``. Do not attach fitted series data or other
        estimator-specific mutable state because the handle may be shared.

        Returns
        -------
        ModelHandle
            Loaded backend objects used by ``_inference``.
        """
        raise NotImplementedError

    def _inference(
        self,
        handle,
        context_y,
        context_X,
        future_X,
        pred_len,
        fh,
        alpha=None,
    ):
        """Run native inference and return a normalized forecast result.

        This required hook translates from sktime's pandas representation to the
        native backend representation. Read backend prediction options from
        ``self.model_spec.predict_extra_kwargs``. It must not format pandas output
        itself.

        Parameters
        ----------
        handle : ModelHandle
            Shared backend objects returned by ``_load_model``.
        context_y : pd.DataFrame
            Target history, shape ``(n_context_timepoints, n_targets)``. Rows are
            chronological and columns correspond to output variables.
        context_X : pd.DataFrame or None
            Exogenous history supplied during fit.
        future_X : pd.DataFrame or None
            Exogenous values supplied for prediction. Its exact row coverage is
            determined by the estimator's X tags and the sktime forecasting API.
        pred_len : int
            Largest relative horizon step. For a standard out-of-sample request,
            generate steps ``1, ..., pred_len`` so sparse steps can be selected by
            the formatter.
        fh : ForecastingHorizon
            Original validated horizon. Inspect this when the backend supports
            sparse, in-sample, or mixed horizons natively.
        alpha : tuple of float or None
            Requested quantile probabilities, or ``None`` for point prediction.

        Returns
        -------
        ForecastResult
            Numeric mean, median, and/or quantile arrays. Each array must have
            shape ``(pred_len, n_targets)`` for a dense horizon or
            ``(len(fh), n_targets)`` in requested horizon order. One-dimensional
            arrays are accepted for univariate targets.

        Notes
        -----
        Quantile output is a mapping ``{probability: values}``; probabilities are
        floats in ``[0, 1]``. At least one of ``mean``, ``median``, or quantile
        ``0.5`` is required for point prediction. Do not mutate shared objects in
        ``handle`` during this method.
        """
        raise NotImplementedError

    def _update_model_spec(self, **changes):
        """Replace fields on the active runtime specification.

        Use this helper for values derived in ``__post_init__`` or
        ``_update_attrs_in_fit``. The frozen specification is replaced rather than
        mutated in place. Unknown fields and invalid extra kwargs are rejected by
        :func:`dataclasses.replace` and ``FoundationModelSpec``.

        Call this before the first handle lookup. Replacing loading settings after
        ``model_handle_`` is attached does not invalidate that existing handle.
        """
        self.model_spec = replace(self.model_spec, **changes)
