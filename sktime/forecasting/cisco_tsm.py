# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Upstream model: splunk/cisco-time-series-model, Apache-2.0 License
# https://github.com/splunk/cisco-time-series-model
"""Cisco Time Series Model (CTSM) forecaster for ``sktime``."""

__author__ = ["vedantag17"]
__all__ = ["CiscoTSMForecaster"]

import numpy as np

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)

# Default quantile levels produced by CTSM. Defined at module level so that
# _DummyCiscoModel can reference them without importing CiscoTSMForecaster.
_DEFAULT_QUANTILES = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    0.95,
    0.99,
]


class CiscoTSMForecaster(BaseFoundationForecaster):
    """Zero-shot univariate forecaster using the Cisco Time Series Model (CTSM).

    CTSM 1.0 is a 250M-parameter, decoder-only transformer foundation model
    developed by Cisco (Splunk) for univariate zero-shot time series
    forecasting [1]_. It uses a multiresolution architecture: internally it
    derives a coarse-resolution context (60 sparser than the input) and a
    fine-resolution context from a single time series, then predicts up to 128
    steps ahead. Long-horizon forecasting beyond 128 steps is supported via
    autoregressive rolling.

    Calling ``fit`` only stores the training context and loads the model
    weights. No model training or fine-tuning is performed.

    Parameters
    ----------
    model_path : str, default="cisco-ai/cisco-time-series-model-1.0"
        HuggingFace repository ID for the CTSM checkpoint.
        Use ``"cisco-ai/cisco-time-series-model-1.0-preview"`` for the
        earlier, larger 500M-parameter preview checkpoint (requires
        ``num_layers=50``).
    num_layers : int, default=25
        Number of transformer layers. Use ``25`` for CTSM 1.0 and
        ``50`` for ``1.0-preview``.
    backend : str, default="cpu"
        Hardware backend: ``"cpu"`` or ``"gpu"``. When set to ``"gpu"``,
        the model is placed on the first available CUDA device. If no GPU
        is found, the package falls back to CPU automatically.
    context_length : int or None, default=None
        Maximum number of most-recent observations to pass as context.
        If ``None``, the full training series (up to the model's internal
        maximum of 30 720 points) is used. Shorter contexts reduce memory
        usage but may degrade forecast quality.
    quantiles : list of float or None, default=None
        Quantile levels pre-computed by the model. If ``None``, the
        official 15-quantile set is used:
        ``[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5,
           0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]``.
        These levels are available via ``predict_quantiles`` /
        ``predict_interval``. Arbitrary ``alpha`` values not in this set
        are handled by linear interpolation over the available levels.
    ignore_deps : bool, default=False
        If ``True``, soft-dependency checks for ``cisco-tsm`` and
        ``torch`` are skipped. Useful for testing the sktime adapter
        contract without the optional packages installed.

    Notes
    -----
    - CTSM is a univariate model. Multivariate targets are not supported.
    - Exogenous variables are not supported.
    - In-sample prediction is not supported.
    - The loaded ``CiscoTsmMR`` object is cached in-process using all settings
      that affect model construction, including the native quantile levels.
    - The model is excluded from the pickle state to keep serialization
      lightweight; it is reloaded transparently on the first ``predict``
      call after unpickling.

    References
    ----------
    .. [1] Liang Gou et al. "Cisco Time Series Model Technical Report."
       arXiv:2511.19841, 2025.
       https://arxiv.org/abs/2511.19841

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.cisco_tsm import CiscoTSMForecaster
    >>> y = load_airline()
    >>> forecaster = CiscoTSMForecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    CiscoTSMForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    >>> pred_int = forecaster.predict_interval(fh=[1, 2, 3],coverage=0.9)#doctest:+SKIP
    """

    _DEFAULT_QUANTILES = _DEFAULT_QUANTILES  # module-level constant
    _uses_torch_inference_context = False

    _tags = {
        # packaging info
        # --------------
        "authors": ["vedantag17"],
        "maintainers": ["vedantag17"],
        "python_dependencies": ["cisco-tsm", "torch"],
        "python_version": ">=3.11,<3.14",
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "None",
        "capability:multivariate": False,
        "capability:exogenous": False,
        "capability:insample": False,
        "capability:missing_values": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "requires-fh-in-fit": False,
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str = "cisco-ai/cisco-time-series-model-1.0",
        num_layers: int = 25,
        backend: str = "cpu",
        context_length=None,
        quantiles=None,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.num_layers = num_layers
        self.backend = backend
        self.context_length = context_length
        self.quantiles = quantiles
        self.ignore_deps = ignore_deps

        native_quantiles = (
            quantiles if quantiles is not None else self._DEFAULT_QUANTILES
        )
        model_spec = FoundationModelSpec(
            model_path=model_path,
            ignore_deps=ignore_deps,
            load_extra_kwargs={
                "num_layers": num_layers,
                "backend": backend,
                "quantiles": native_quantiles,
            },
            predict_extra_kwargs={"context_length": context_length},
        )
        super().__init__(model_spec=model_spec)

    def _load_model(self):
        """Instantiate the CiscoTsmMR model for the shared handle cache.

        When ``ignore_deps=True`` a lightweight built-in dummy is returned so
        that ``check_estimator`` and unit tests can exercise the adapter
        contract without installing ``cisco-tsm``.
        """
        model_spec = self.model_spec
        load_kwargs = model_spec.load_extra_kwargs
        if model_spec.ignore_deps:
            model = _DummyCiscoModel(
                horizon_fill=0.0,
                quantiles=load_kwargs["quantiles"],
            )
            return ModelHandle(model=model)

        from sktime.utils.dependencies import _check_soft_dependencies

        _check_soft_dependencies("cisco-tsm", "torch", severity="error")
        from cisco_tsm import CiscoTsmMR, TimesFmCheckpoint, TimesFmHparams

        hparams = TimesFmHparams(
            num_layers=load_kwargs["num_layers"],
            use_positional_embedding=False,
            backend=load_kwargs["backend"],
            quantiles=load_kwargs["quantiles"],
        )
        checkpoint = TimesFmCheckpoint(
            huggingface_repo_id=model_spec.model_path,
        )
        model = CiscoTsmMR(hparams=hparams, checkpoint=checkpoint)
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
        """Forecast from the univariate context and interpolate quantiles."""
        context_length = self.model_spec.predict_extra_kwargs["context_length"]
        context = context_y.iloc[:, 0].to_numpy(dtype=np.float32)
        if context_length is not None and len(context) > context_length:
            context = context[-context_length:]

        forecast = handle.model.forecast(context, horizon_len=pred_len)[0]
        mean = np.asarray(forecast["mean"], dtype=np.float32)
        if alpha is None:
            return ForecastResult(mean=mean)

        native_quantiles = forecast["quantiles"]
        native_keys = sorted(native_quantiles)
        native_levels = np.asarray(native_keys, dtype=float)
        native_values = np.stack(
            [native_quantiles[key] for key in native_keys],
            axis=0,
        )
        quantiles = {
            float(quantile): np.asarray(
                [
                    np.interp(quantile, native_levels, native_values[:, step])
                    for step in range(pred_len)
                ],
                dtype=np.float32,
            )
            for quantile in alpha
        }
        return ForecastResult(mean=mean, quantiles=quantiles)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return. For use in tests.
            No special values are currently reserved for forecasters.

        Returns
        -------
        params : list of dict
            Each dict is a valid constructor argument set for testing.
        """
        return [
            {
                "model_path": "cisco-ai/cisco-time-series-model-1.0",
                "ignore_deps": True,
            },
            {
                "model_path": "cisco-ai/cisco-time-series-model-1.0",
                "context_length": 64,
                "ignore_deps": True,
            },
            {
                "model_path": "cisco-ai/cisco-time-series-model-1.0",
                "quantiles": [0.1, 0.5, 0.9],
                "ignore_deps": True,
            },
        ]


class _DummyCiscoModel:
    """Lightweight stub returned by ``_load_model`` when ``ignore_deps=True``.

    This zero-dependency class mimics the ``CiscoTsmMR.forecast`` interface
    and returns a constant array so that ``check_estimator`` and unit tests
    can exercise the full fit/predict lifecycle without ``cisco-tsm`` installed.

    Parameters
    ----------
    horizon_fill : float, default=0.0
        Constant value returned for every forecast step.
    """

    def __init__(self, horizon_fill: float = 0.0, quantiles=None):
        self.horizon_fill = horizon_fill
        self.quantiles = _DEFAULT_QUANTILES if quantiles is None else list(quantiles)

    def forecast(self, series, horizon_len):
        """Return constant forecasts matching the CiscoTsmMR output format."""
        mean = np.full(horizon_len, self.horizon_fill, dtype=np.float32)
        # Provide the same 15 native quantile levels as the real model so that
        # _predict_quantiles can interpolate without special-casing the dummy.
        # Reference the class-level constant directly to avoid a circular import.
        quantiles = {
            q: np.full(horizon_len, self.horizon_fill, dtype=np.float32)
            for q in self.quantiles
        }
        return [{"mean": mean, "quantiles": quantiles}]
