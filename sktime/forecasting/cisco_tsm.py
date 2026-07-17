# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# Upstream model: splunk/cisco-time-series-model, Apache-2.0 License
# https://github.com/splunk/cisco-time-series-model
"""Cisco Time Series Model (CTSM) forecaster for ``sktime``."""

__author__ = ["vedantag17"]
__all__ = ["CiscoTSMForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton

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


class CiscoTSMForecaster(BaseForecaster):
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
    - The loaded ``CiscoTsmMR`` object is cached in-process via a
      multiton keyed on ``(model_path, num_layers, backend)`` to avoid
      redundant model loading across multiple estimator instances.
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

    _tags = {
        # packaging info
        # --------------
        "authors": ["vedantag17"],
        "maintainers": ["vedantag17"],
        "python_dependencies": ["cisco-tsm", "torch"],
        "python_version": ">=3.11,<3.14",
        # estimator type
        # --------------
        "y_inner_mtype": "pd.Series",
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

        # leave this as is
        super().__init__()

    def __getstate__(self):
        """Return state for pickling, excluding the heavy model object."""
        state = self.__dict__.copy()
        state["_model_"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled dict."""
        self.__dict__.update(state)

    def _get_unique_key(self) -> str:
        """Build a deterministic cache key for the multiton model loader."""
        return (
            f"model_path={self.model_path}|"
            f"num_layers={self.num_layers}|"
            f"backend={self.backend}"
        )

    def _load_model(self):
        """Instantiate and return a cached CiscoTsmMR model.

        When ``ignore_deps=True`` a lightweight built-in dummy is returned so
        that ``check_estimator`` and unit tests can exercise the adapter
        contract without installing ``cisco-tsm``.
        """
        if self.ignore_deps:
            return _DummyCiscoModel(horizon_fill=0.0)

        from sktime.utils.dependencies import _check_soft_dependencies

        _check_soft_dependencies("cisco-tsm", "torch", severity="error")

        _quantiles = (
            self.quantiles if self.quantiles is not None else self._DEFAULT_QUANTILES
        )
        return _CachedCiscoTSM(
            key=self._get_unique_key(),
            model_path=self.model_path,
            num_layers=self.num_layers,
            backend=self.backend,
            quantiles=_quantiles,
        ).load()

    def _ensure_model_loaded(self):
        """Reload the model lazily, e.g. after unpickling."""
        if not hasattr(self, "_model_") or self._model_ is None:
            if self._is_fitted:
                self._model_ = self._load_model()

    def _fit(self, y, X=None, fh=None):
        """Load the model and store training context.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in ``_``.

        Parameters
        ----------
        y : pd.Series
            Guaranteed univariate time series (``capability:multivariate`` is
            ``False``). Used as the forecasting context.
        X : ignored
            Exogenous time series are not supported.
        fh : ForecastingHorizon or None
            Not required at fit time (``requires-fh-in-fit`` is ``False``).

        Returns
        -------
        self : reference to self
        """
        self._model_ = self._load_model()

        # Store the context array; clip to context_length if specified.
        context = y.to_numpy(dtype=np.float32)
        if self.context_length is not None and len(context) > self.context_length:
            context = context[-self.context_length :]

        self._context_ = context
        self._y_name_ = y.name

        return self

    def _predict(self, fh, X=None):
        """Forecast future values for the given horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be ``"fitted"``.

        Accesses in self:
            ``_model_``, ``_context_``, ``_y_name_``, ``cutoff``

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon. Only out-of-sample horizons are supported.
        X : ignored

        Returns
        -------
        y_pred : pd.Series
            Point forecasts with absolute index matching ``fh``.
        """
        self._ensure_model_loaded()

        fh_relative = fh.to_relative(self.cutoff)
        horizon_len = int(max(fh_relative._values))

        # Run CTSM inference.
        # model.forecast returns list[dict] with keys 'mean' (np.ndarray of
        # shape (horizon_len,)) and 'quantiles' (dict).
        forecast_preds = self._model_.forecast(
            self._context_,
            horizon_len=horizon_len,
        )
        mean_forecast = forecast_preds[0]["mean"]  # (horizon_len,)

        # Build output with the canonical sktime index idiom.
        row_idx = fh.to_absolute_index(self.cutoff)

        # fh_relative is 1-based; map to 0-based positions in mean_forecast.
        fh_vals = np.asarray(fh_relative._values, dtype=int) - 1

        y_pred = pd.Series(
            mean_forecast[fh_vals],
            index=row_idx,
            name=self._y_name_,
        )
        return y_pred

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be ``"fitted"``.

        Accesses in self:
            ``_model_``, ``_context_``, ``_y_name_``, ``cutoff``

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : ignored
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.
            Values not in the model's native quantile set are linearly
            interpolated from the surrounding available levels.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        self._ensure_model_loaded()

        fh_relative = fh.to_relative(self.cutoff)
        horizon_len = int(max(fh_relative._values))
        fh_vals = np.asarray(fh_relative._values, dtype=int) - 1

        forecast_preds = self._model_.forecast(
            self._context_,
            horizon_len=horizon_len,
        )
        quantiles_dict = forecast_preds[0]["quantiles"]  # dict[float, np.ndarray]

        # Sort available native levels for interpolation.
        native_levels = sorted(quantiles_dict.keys())
        # Stack into (n_native_quantiles, horizon_len) array.
        native_matrix = np.stack(
            [quantiles_dict[q] for q in native_levels], axis=0
        )  # shape: (n_q, horizon_len)

        var_name = self._y_name_ if self._y_name_ is not None else 0
        row_idx = fh.to_absolute_index(self.cutoff)
        cols_idx = pd.MultiIndex.from_product([[var_name], alpha])
        pred_quantiles = pd.DataFrame(index=row_idx, columns=cols_idx, dtype=float)

        native_arr = np.array(native_levels, dtype=float)
        for a in alpha:
            # Linear interpolation across quantile dimension at each horizon step.
            interp_vals = np.array(
                [np.interp(a, native_arr, native_matrix[:, t]) for t in fh_vals],
                dtype=np.float32,
            )
            pred_quantiles[(var_name, a)] = interp_vals

        return pred_quantiles

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        State required:
            Requires state to be "fitted".

        Accesses in self:
            _model_, _context_, _y_name_, cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : ignored
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : skpro HistogramQPD
            predictive distribution
        """
        self._ensure_model_loaded()

        fh_relative = fh.to_relative(self.cutoff)
        horizon_len = int(max(fh_relative._values))
        fh_vals = np.asarray(fh_relative._values, dtype=int) - 1

        forecast_preds = self._model_.forecast(
            self._context_,
            horizon_len=horizon_len,
        )
        quantiles_dict = forecast_preds[0]["quantiles"]  # dict[float, np.ndarray]

        # Sort available native levels for the QPD knots.
        knots = sorted(quantiles_dict.keys())
        # Stack into (n_knots, horizon_len) array.
        native_matrix = np.stack(
            [quantiles_dict[q] for q in knots], axis=0
        )  # shape: (n_knots, horizon_len)

        # Slice to select the target horizon points.
        selected_matrix = native_matrix[:, fh_vals]  # shape: (n_knots, len(fh_vals))

        var_name = self._y_name_ if self._y_name_ is not None else 0
        pred_index = fh.to_absolute_index(self.cutoff)
        row_idx = pd.MultiIndex.from_product([knots, pred_index])

        # For a univariate forecaster, the DataFrame columns is a single element.
        columns = [var_name]

        # Reshape data to (len(knots) * len(pred_index), 1) matching the product index.
        # Outer level: knots (quantiles), Inner level: pred_index (time indices).
        # Since row-major order is used, the inner level (time) varies fastest.
        q_df = pd.DataFrame(
            data=selected_matrix.reshape(-1, 1),
            index=row_idx,
            columns=columns,
            dtype=float,
        )

        from skpro.distributions import HistogramQPD

        return HistogramQPD(q_df, tails="mass", index=pred_index, columns=columns)

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

    def __init__(self, horizon_fill: float = 0.0):
        self.horizon_fill = horizon_fill

    def forecast(self, series, horizon_len):
        """Return constant forecasts matching the CiscoTsmMR output format."""
        mean = np.full(horizon_len, self.horizon_fill, dtype=np.float32)
        # Provide the same 15 native quantile levels as the real model so that
        # _predict_quantiles can interpolate without special-casing the dummy.
        # Reference the class-level constant directly to avoid a circular import.
        quantiles = {
            q: np.full(horizon_len, self.horizon_fill, dtype=np.float32)
            for q in _DEFAULT_QUANTILES
        }
        return [{"mean": mean, "quantiles": quantiles}]


@_multiton
class _CachedCiscoTSM:
    """In-process cache for a loaded ``CiscoTsmMR`` model.

    A single ``CiscoTsmMR`` instance is shared across all ``CiscoTSMForecaster``
    objects that use the same ``(model_path, num_layers, backend)`` combination.
    This avoids redundant HuggingFace downloads and memory duplication.

    Parameters
    ----------
    key : str
        Multiton key (used externally for lookup).
    model_path : str
        HuggingFace repository ID.
    num_layers : int
        Number of transformer layers.
    backend : str
        ``"cpu"`` or ``"gpu"``.
    quantiles : list of float
        Quantile levels for the model.
    """

    def __init__(self, key, model_path, num_layers, backend, quantiles):
        self.key = key
        self.model_path = model_path
        self.num_layers = num_layers
        self.backend = backend
        self.quantiles = quantiles
        self._model = None

    def load(self):
        """Instantiate and return the cached ``CiscoTsmMR`` model.

        On first call the model is downloaded from HuggingFace and
        initialised; subsequent calls return the cached instance.

        Returns
        -------
        model : CiscoTsmMR
        """
        if self._model is not None:
            return self._model

        from cisco_tsm import CiscoTsmMR, TimesFmCheckpoint, TimesFmHparams

        hparams = TimesFmHparams(
            num_layers=self.num_layers,
            use_positional_embedding=False,
            backend=self.backend,
            quantiles=self.quantiles,
        )
        ckpt = TimesFmCheckpoint(huggingface_repo_id=self.model_path)
        self._model = CiscoTsmMR(hparams=hparams, checkpoint=ckpt)
        return self._model
