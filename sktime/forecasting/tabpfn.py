# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TabPFN Forecaster for sktime.

Wraps the TabPFN-TS pipeline as an sktime-compatible univariate forecaster.
Soft dependencies: tabpfn_time_series.
"""

__author__ = ["lucifer4073"]
__all__ = ["TabPFNForecaster"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster


class TabPFNForecaster(BaseForecaster):
    """TabPFN-based zero-shot forecaster for univariate time series.

    Wraps ``TabPFNTSPipeline`` from the ``tabpfn_time_series`` package as an
    sktime-compatible forecaster.  The underlying model is a transformer
    pre-trained on synthetic time series; no training data is required to fit
    a useful prior -- fitting merely stores the context series.

    TabPFN can run inference either against the hosted cloud API
    (``tabpfn_mode="client"``) or against a locally-loaded checkpoint
    (``tabpfn_mode="local"``).

    Parameters
    ----------
    max_context_length : int, default=4096
        Maximum number of historical observations passed to TabPFN as context.
        Longer contexts improve accuracy but increase memory and compute cost.
        Must be a positive integer.
    tabpfn_mode : str, default="client"
        Inference backend.  ``"client"`` routes predictions through the
        TabPFN cloud API; ``"local"`` runs inference on the local machine
        using a downloaded model checkpoint.
    tabpfn_output_selection : str, default="median"
        Determines which probabilistic output is returned as the point
        forecast.  Supported values are ``"median"`` and ``"mean"``.
    ignore_future_covariates_if_missing : bool, default=False
        If ``True``, covariates present in the context but absent from the
        prediction horizon are silently ignored rather than raising an error.

    Attributes
    ----------
    _pipeline_ : TabPFNTSPipeline
        Fitted pipeline instance, available after ``fit``.
    _y_context : pd.Series
        Copy of the training series ``y``, stored for use during prediction.
    _X_context : pd.DataFrame or None
        Copy of the training exogenous data ``X``, or ``None`` when not
        provided.
    _cutoff_val : index element
        Last index value of the training series, used to resolve relative
        forecasting horizons to absolute timestamps.

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

    Index types ``pd.PeriodIndex`` and ``pd.DatetimeIndex`` are both handled
    transparently.  The returned prediction index always matches the type of
    the training index.

    References
    ----------
    .. [1] Müller, S. et al., "TabPFN: A Transformer That Solves Small
       Tabular Classification Problems in a Second", ICLR 2023.

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
        "capability:missing_values": True,
        "capability:insample": False,
        "python_dependencies": ["tabpfn_time_series"],
        "authors": ["sktime-contributor"],
    }

    def __init__(
        self,
        max_context_length: int = 4096,
        tabpfn_mode: str = "client",
        tabpfn_output_selection: str = "median",
        ignore_future_covariates_if_missing: bool = False,
    ):
        self.max_context_length = max_context_length
        self.tabpfn_mode = tabpfn_mode
        self.tabpfn_output_selection = tabpfn_output_selection
        self.ignore_future_covariates_if_missing = ignore_future_covariates_if_missing

        super().__init__()

    def _make_context_df(self, y: pd.Series, X: "pd.DataFrame | None") -> pd.DataFrame:
        """Build the flat context DataFrame expected by ``predict_df``.

        Combines the target series and optional covariates into a single
        DataFrame with columns ``timestamp``, ``target``, and any covariate
        columns.  The ``item_id`` column is omitted; ``predict_df`` inserts a
        dummy value for single-series inputs automatically.

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

        Creates a DataFrame covering the requested prediction horizon with
        optional covariate values.  The ``target`` column is intentionally
        omitted; ``_preprocess_future`` inside the pipeline inserts NaN
        values as placeholders.

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

        Handles ``pd.PeriodIndex``, ``pd.DatetimeIndex``, and plain
        ``pd.Index`` without raising.  For ``pd.DatetimeIndex`` where
        ``freq`` is ``None``, ``pd.infer_freq`` is used as a fallback.

        Returns
        -------
        freq : offset alias str, pd.DateOffset, or None
            The inferred frequency, or ``None`` if it cannot be determined
            (e.g. for integer or object indices).
        """
        idx = self._y_context.index
        if isinstance(idx, pd.PeriodIndex):
            return idx.freq
        elif isinstance(idx, pd.DatetimeIndex):
            return idx.freq or pd.infer_freq(idx)
        return None

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to training data.

        Instantiates the ``TabPFNTSPipeline`` and stores the training series
        and covariates for use during prediction.  TabPFN is a zero-shot
        model, so no gradient updates or parameter estimation occur here.

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
        from tabpfn_time_series import (
            TABPFN_DEFAULT_CONFIG,
            TabPFNMode,
            TabPFNTSPipeline,
        )

        mode_map = {
            "client": TabPFNMode.CLIENT,
            "local": TabPFNMode.LOCAL,
        }

        self._pipeline_ = TabPFNTSPipeline(
            max_context_length=self.max_context_length,
            tabpfn_mode=mode_map.get(self.tabpfn_mode, TabPFNMode.CLIENT),
            tabpfn_output_selection=self.tabpfn_output_selection,
            tabpfn_model_config=TABPFN_DEFAULT_CONFIG,
        )

        self._y_context = y.copy()
        self._X_context = X.copy() if X is not None else None
        self._cutoff_val = y.index[-1]

        return self

    def _predict(self, fh, X=None):
        """Generate point forecasts for the requested horizon.

        Converts the stored context and the requested horizon to the flat
        DataFrame format required by ``TabPFNTSPipeline.predict_df``, calls
        the pipeline, and reformats the output to a ``pd.Series`` with the
        same index type and name as the training series.

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
        fh_abs = fh.to_absolute(self._cutoff_val)

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
            Name of the parameter set to return.  Currently only
            ``"default"`` is supported.

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
                "ignore_future_covariates_if_missing": False,
            },
            {
                "max_context_length": 2048,
                "tabpfn_mode": "local",
                "tabpfn_output_selection": "median",
                "ignore_future_covariates_if_missing": True,
            },
            {
                "max_context_length": 4096,
                "tabpfn_mode": "local",
                "tabpfn_output_selection": "mean",
                "ignore_future_covariates_if_missing": False,
            },
        ]
        return params_list
