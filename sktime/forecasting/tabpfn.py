# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TabPFN Forecaster for sktime."""

__author__ = ["lucifer4073"]
__all__ = ["TabPFNForecaster"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton

if _check_soft_dependencies("tabpfn_time_series", severity="none"):
    from tabpfn_time_series import (
        TABPFN_DEFAULT_CONFIG,
        TabPFNMode,
    )
    from tabpfn_time_series import (
        TabPFNTSPipeline as _TabPFNTSPipeline,
    )
else:

    class TabPFNMode:  # type: ignore[no-redef]
        """Stub for ``TabPFNMode`` when ``tabpfn_time_series`` is not installed."""

        CLIENT = "client"
        LOCAL = "local"

    TABPFN_DEFAULT_CONFIG = {}  # type: ignore[assignment]

    class _TabPFNTSPipeline:  # type: ignore[no-redef]
        """Stub for ``TabPFNTSPipeline`` when ``tabpfn_time_series``is not installed."""


class TabPFNForecaster(BaseForecaster):
    """TabPFN-based zero-shot forecaster for univariate time series.

    Wraps ``TabPFNTSPipeline`` from the ``tabpfn_time_series`` package as an
    sktime-compatible forecaster.

    Parameters
    ----------
    max_context_length : int, default=4096
        Maximum number of historical observations passed to TabPFN as context.
        Longer contexts improve accuracy but increase memory and compute cost.
        Must be a positive integer.
    tabpfn_mode : str, default="local"
        Inference backend.  ``"client"`` routes predictions through the
        TabPFN cloud API; ``"local"`` runs inference on the local machine
        using a downloaded model checkpoint.
    tabpfn_output_selection : str, default="median"
        Determines which probabilistic output is returned as the point
        forecast.  Supported values are ``"median"`` and ``"mean"``.

    Attributes
    ----------
    _pipeline_ : TabPFNTSPipeline
        Shared (cached) pipeline instance, available after ``fit``.
    _y_context : pd.Series
        Copy of the training series ``y``, stored for use during prediction.
    _X_context : pd.DataFrame or None
        Copy of the training exogenous data ``X``, or ``None`` when not
        provided.

    See Also
    --------
    sktime.forecasting.base.BaseForecaster :
        Base class for all sktime forecasters.

    Notes
    -----
    The forecaster relies on the soft dependency ``tabpfn_time_series``.
    Install it with ``pip install tabpfn-time-series`` before use.

    Only univariate forecasting is supported (``scitype:y = "univariate"``).
    In-sample predictions are not supported (``capability:insample = False``).

    References
    ----------
    .. [1] https://github.com/PriorLabs/tabpfn-time-series
    .. [2] Hoo, L.S.B. et al., "TabPFN-TS: Zero-shot Time Series
       Forecasting with TabPFNv2", arXiv preprint arXiv:2501.02945, 2025.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tabpfn import TabPFNForecaster
    >>> y = load_airline()
    >>> forecaster = TabPFNForecaster(tabpfn_mode="local")
    >>> forecaster.fit(y)
    TabPFNForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "capability__pred_int": False,
        "capability__pred_int__insample": False,
        "capability__categorical_in_X": True,
        "capability__pretrain": True,
        "capability:missing_values": True,
        "capability:insample": False,
        "property__randomness": "probabilistic",
        "python_dependencies": ["tabpfn_time_series"],
        "authors": ["lucifer4073"],
        "maintainers": ["lucifer4073"],
        "python_version": ">=3.10",
        "tests:skip_by_name": [
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    }

    def __init__(
        self,
        max_context_length: int = 4096,
        tabpfn_mode: str = "local",
        tabpfn_output_selection: str = "median",
    ):
        self.max_context_length = max_context_length
        self.tabpfn_mode = tabpfn_mode
        self.tabpfn_output_selection = tabpfn_output_selection

        super().__init__()

    def _get_pipeline_kwargs(self) -> dict:
        """Return the kwargs that fully characterise the ``TabPFNTSPipeline``.

        Returns
        -------
        dict
            Keyword arguments used both to construct the pipeline and to
            derive the multiton cache key.
        """
        return {
            "max_context_length": self.max_context_length,
            "tabpfn_mode": self.tabpfn_mode,
            "tabpfn_output_selection": self.tabpfn_output_selection,
        }

    def _get_unique_key(self) -> str:
        """Derive a stable string key for the multiton cache.

        Returns
        -------
        str
            Sorted, stringified representation of ``_get_pipeline_kwargs()``.
        """
        return str(sorted(self._get_pipeline_kwargs().items()))

    def _load_pipeline(self) -> _TabPFNTSPipeline:
        """Obtain the shared ``TabPFNTSPipeline`` from the multiton cache.

        Returns
        -------
        _TabPFNTSPipeline
            A ready-to-use pipeline instance.
        """
        return _CachedTabPFN(
            key=self._get_unique_key(),
            pipeline_kwargs=self._get_pipeline_kwargs(),
        ).load_pipeline()

    def _ensure_pipeline_loaded(self):
        """Re-acquire the pipeline if it was lost (e.g., after unpickling)."""
        if not hasattr(self, "_pipeline_") or self._pipeline_ is None:
            if getattr(self, "_is_fitted", False):
                self._pipeline_ = self._load_pipeline()

    def __getstate__(self):
        """Return a pickleable state dict."""
        state = self.__dict__.copy()
        state["_pipeline_"] = None
        return state

    def __setstate__(self, state):
        """Restore instance state and mark pipeline for lazy re-loading."""
        self.__dict__.update(state)

    def _make_context_df(self, y: pd.Series, X: "pd.DataFrame | None") -> pd.DataFrame:
        """Build the flat context DataFrame expected by ``predict_df``.

        Parameters
        ----------
        y : pd.Series
            Historical target series with a ``pd.PeriodIndex`` or
            ``pd.DatetimeIndex``.
        X : pd.DataFrame or None
            Historical exogenous covariates aligned to the index of ``y``, or
            ``None`` when no covariates are available.

        Returns
        -------
        pd.DataFrame
            Flat DataFrame with columns ``["timestamp", "target", ...]``
            where ``timestamp`` holds ``datetime64`` values.
        """
        df = y.rename("target").to_frame()
        if X is not None:
            df = df.join(X, how="left")
        df.index.name = "timestamp"
        if isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.to_timestamp()
        return df.reset_index()

    def _make_future_df(self, fh_abs, X: "pd.DataFrame | None") -> pd.DataFrame:
        """Build the flat future DataFrame expected by ``predict_df``.

        Parameters
        ----------
        fh_abs : ForecastingHorizon
            Absolute forecasting horizon whose ``to_pandas`` method returns a
            ``pd.PeriodIndex`` or ``pd.DatetimeIndex``.
        X : pd.DataFrame or None
            Future exogenous covariates aligned to the horizon index, or
            ``None`` when not available.

        Returns
        -------
        pd.DataFrame
            Flat DataFrame with columns ``["timestamp", ...]`` where
            ``timestamp`` holds ``datetime64`` values.  No ``target`` column
            is present.
        """
        fh_indices = fh_abs.to_pandas()
        timestamps = pd.DataFrame(index=fh_indices)
        if X is not None:
            timestamps = timestamps.join(X, how="left")
        timestamps.index.name = "timestamp"
        if isinstance(timestamps.index, pd.PeriodIndex):
            timestamps.index = timestamps.index.to_timestamp()
        return timestamps.reset_index()

    def _get_freq(self):
        """Safely infer the time-series frequency from the context index.

        Returns
        -------
        freq : offset alias str, pd.DateOffset, or None
            The inferred frequency, or ``None`` if it cannot be determined.
        """
        idx = self._y_context.index
        if isinstance(idx, pd.PeriodIndex):
            return idx.freq
        elif isinstance(idx, pd.DatetimeIndex):
            return idx.freq or pd.infer_freq(idx)
        return None

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to fit the forecaster on.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series covariates.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon; not required at fit time.

        Returns
        -------
        self : TabPFNForecaster
            Reference to the fitted forecaster instance.
        """
        self._pipeline_ = self._load_pipeline()

        self._y_context = y.copy()
        self._X_context = X.copy() if X is not None else None

        return self

    def _predict(self, fh, X=None):
        """Generate point forecasts for the requested horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon, either relative or absolute.
        X : pd.DataFrame, optional (default=None)
            Future exogenous covariates aligned to the horizon.

        Returns
        -------
        y_pred : pd.Series
            Predicted values indexed by the absolute forecasting horizon.
            The series name matches the name of the training series ``y``.
        """
        self._ensure_pipeline_loaded()

        fh_abs = fh.to_absolute(self.cutoff)

        context_df = self._make_context_df(self._y_context, self._X_context)
        future_df = self._make_future_df(fh_abs, X)

        preds_df = self._pipeline_.predict_df(
            context_df=context_df,
            future_df=future_df,
        )

        preds_df = preds_df.reset_index(level="item_id", drop=True)

        y_pred = preds_df["target"]
        y_pred.name = self._y_context.name

        freq = self._get_freq()
        if isinstance(self._y_context.index, pd.PeriodIndex) and freq is not None:
            y_pred.index = y_pred.index.to_period(freq=freq)

        y_pred = y_pred.reindex(fh_abs.to_pandas())

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the parameter set to return.

        Returns
        -------
        params : list of dict
            List of dictionaries, each containing a valid parameter
            combination for instantiating the estimator in tests.
        """
        params_list = [
            {
                "max_context_length": 512,
                "tabpfn_mode": "local",
                "tabpfn_output_selection": "median",
            },
            {
                "max_context_length": 2048,
                "tabpfn_mode": "local",
                "tabpfn_output_selection": "median",
            },
            {
                "max_context_length": 4096,
                "tabpfn_mode": "local",
                "tabpfn_output_selection": "mean",
            },
        ]
        return params_list


@_multiton
class _CachedTabPFN:
    """Cached ``TabPFNTSPipeline``, ensuring one instance per unique configuration."""

    def __init__(self, key: str, pipeline_kwargs: dict):
        self.key = key
        self.pipeline_kwargs = pipeline_kwargs
        self._pipeline = None

    def load_pipeline(self):
        """Return the shared pipeline, constructing it on first call.

        Returns
        -------
        TabPFNTSPipeline
            A fully initialised ``TabPFNTSPipeline`` ready for ``predict_df``.
        """
        if self._pipeline is not None:
            return self._pipeline

        from tabpfn_time_series import (
            TABPFN_DEFAULT_CONFIG,
            TabPFNMode,
            TabPFNTSPipeline,
        )

        mode_map = {
            "client": TabPFNMode.CLIENT,
            "local": TabPFNMode.LOCAL,
        }

        tabpfn_mode_str = self.pipeline_kwargs["tabpfn_mode"]

        if tabpfn_mode_str not in mode_map:
            raise ValueError(
                f"tabpfn_mode must be 'local' or 'client', got '{tabpfn_mode_str}'"
            )

        allowed_output_selections = {"median", "mean"}
        tabpfn_output_selection = self.pipeline_kwargs["tabpfn_output_selection"]
        max_context_length = self.pipeline_kwargs["max_context_length"]

        if tabpfn_output_selection not in allowed_output_selections:
            raise ValueError(
                f"Invalid `tabpfn_output_selection`: {tabpfn_output_selection!r}. "
                f"Supported values are {sorted(allowed_output_selections)}."
            )
        if not isinstance(max_context_length, int) or max_context_length <= 0:
            raise ValueError(
                f"max_context_length must be a positive integer, "
                f"but got {max_context_length!r}."
            )

        self._pipeline = TabPFNTSPipeline(
            max_context_length=max_context_length,
            tabpfn_mode=mode_map.get(tabpfn_mode_str, TabPFNMode.LOCAL),
            tabpfn_output_selection=tabpfn_output_selection,
            tabpfn_model_config=TABPFN_DEFAULT_CONFIG,
        )
        return self._pipeline
