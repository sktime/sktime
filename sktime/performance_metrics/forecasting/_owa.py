"""Overall Weighted Average (OWA) metric for forecasting."""

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
)
from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric

__author__ = ["jgyasu"]
__all__ = ["OverallWeightedAverage", "m4_naive2_forecast", "m4_seasonality_test"]


def m4_seasonality_test(y, sp):
    """Test for seasonality using the M4 competition 90% ACF criterion.

    Implements ``SeasonalityTest`` from the official M4 benchmarks code:
    https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R

    Parameters
    ----------
    y : array-like, 1D
        In-sample time series values.
    sp : int
        Seasonal period to test (M4 frequency ``m``).

    Returns
    -------
    bool
        True if seasonality is detected at lag ``sp``.
    """
    from statsmodels.tsa.stattools import acf

    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if sp <= 1 or n < 3 * sp:
        return False

    xacf = acf(y, nlags=sp, fft=False)[1:]
    clim = (
        1.645 / np.sqrt(n) * np.sqrt(np.cumsum(np.concatenate([[1.0], 2.0 * xacf**2])))
    )
    test_seasonal = abs(xacf[sp - 1]) > clim[sp - 1]

    if np.isnan(test_seasonal):
        return False
    return bool(test_seasonal)


def m4_naive2_forecast(y, fh, sp):
    """Generate Naive-2 forecasts as defined in the M4 competition.

    Implements the Naive-2 benchmark from the official M4 benchmarks code:
    https://github.com/Mcompetitions/M4-methods/blob/master/Benchmarks%20and%20Evaluation.R

    Parameters
    ----------
    y : array-like, 1D
        In-sample time series values.
    fh : int
        Forecast horizon length.
    sp : int
        Seasonal period (M4 frequency ``m``).

    Returns
    -------
    np.ndarray
        Naive-2 point forecasts of length ``fh``.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    y = np.asarray(y, dtype=float).ravel()
    fh = int(fh)

    if sp > 1 and m4_seasonality_test(y, sp):
        decomposition = seasonal_decompose(
            y,
            model="multiplicative",
            period=sp,
            extrapolate_trend="freq",
        )
        seasonal = np.asarray(decomposition.seasonal, dtype=float)
        deseasonalized = y / seasonal
        seasonal_out = np.tile(seasonal[-sp:], int(np.ceil(fh / sp)))[:fh]
    else:
        deseasonalized = y
        seasonal_out = np.ones(fh, dtype=float)

    return np.repeat(deseasonalized[-1], fh) * seasonal_out


class OverallWeightedAverage(BaseForecastingErrorMetric):
    r"""Overall Weighted Average (OWA) metric as used in the M4 competition.

    The OWA metric combines MASE and sMAPE, each normalized by the corresponding
    error of a Naive-2 benchmark forecaster:

    .. math::
        \text{OWA} =
        0.5 \left(
            \frac{\text{MASE}}{\text{MASE}_{\text{Naive2}}}
            +
            \frac{\text{sMAPE}}{\text{sMAPE}_{\text{Naive2}}}
        \right)

    Lower values indicate better forecasting performance.

    For a single series, MASE and sMAPE are averaged over the forecast horizon
    before forming the ratio. When evaluating panel or hierarchical data with
    ``multilevel="uniform_average"`` (default), MASE and sMAPE are averaged
    across series first and the OWA ratio is computed once, matching the M4
    competition aggregation.

    The Naive-2 benchmark follows the official M4 implementation:

    * Apply a 90% autocorrelation seasonality test at lag ``sp``
    * If seasonal, apply classical multiplicative decomposition, forecast the
      deseasonalized series with a random walk (last value), and reseasonalize
      using the last seasonal period
    * If not seasonal, use a random walk (last value) on the original series

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity (M4 frequency ``m``). Use 12 for monthly, 4 for
        quarterly, 24 for hourly, and 1 for yearly, weekly, or daily data.

    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
        Absolute values smaller than eps are replaced by eps.
        If None, defaults to np.finfo(np.float64).eps

    multioutput : 'uniform_average' (default), 1D array-like, or 'raw_values'
        Whether and how to aggregate metric for multivariate (multioutput) data.

        * If ``'uniform_average'`` (default),
          errors of all outputs are averaged with uniform weight.
        * If 1D array-like, errors are averaged across variables,
          with values used as averaging weights (same order).
        * If ``'raw_values'``,
          does not average across variables (outputs), per-variable errors are returned.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        How to aggregate the metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          MASE and sMAPE are averaged across series and OWA is computed once,
          as in the M4 competition.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Controls averaging over time points in direct call to metric object.

        * If ``False`` (default),
          direct call to the metric object averages over time points,
          equivalent to a call of the ``evaluate`` method.
        * If ``True``, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.

    References
    ----------
    Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020).
    The M4 Competition: 100,000 time series and 61 forecasting methods.
    International Journal of Forecasting, 36(1), 54-74.
    https://doi.org/10.1016/j.ijforecast.2019.04.014

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import OverallWeightedAverage
    >>> import numpy as np
    >>> y_true = np.array([100.05, 100.0, 100.15])
    >>> y_pred = np.array([100.0, 100.05, 100.0])
    >>> y_train = np.array([100.0, 100.1, 100.0, 100.2, 100.1, 100.0])
    >>> metric = OverallWeightedAverage(sp=1)
    >>> metric(y_true, y_pred, y_train=y_train)  # doctest: +SKIP
    np.float64(1.2500468574284465)
    """

    _tags = {
        "requires-y-train": True,
        "python_dependencies": ["statsmodels"],
        "tests:skip_by_name": ["test_uniform_average_time"],
    }

    def __init__(
        self,
        sp=1,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
        eps=None,
    ):
        self.sp = sp
        self.eps = eps
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _fh_index(self, y_true):
        """Forecast horizon index aligned with ``y_true`` rows."""
        index = y_true.index
        if hasattr(index, "nlevels") and index.nlevels > 1:
            return index.get_level_values(-1).unique()
        return index

    def _predict_naive2(self, y_train, y_true):
        """Fit Naive-2 on training data and predict for the test horizon."""
        fh_index = self._fh_index(y_true)
        fh = len(fh_index)

        if isinstance(y_true, pd.DataFrame):
            if isinstance(y_train, pd.DataFrame):
                train_df = y_train
            elif isinstance(y_train, pd.Series):
                train_df = y_train.to_frame()
            else:
                train_df = pd.DataFrame(
                    {y_true.columns[0]: np.asarray(y_train, dtype=float).ravel()}
                )

            forecasts = {
                col: m4_naive2_forecast(train_df[col].to_numpy(), fh, self.sp)
                for col in y_true.columns
            }
            return pd.DataFrame(forecasts, index=fh_index)

        if isinstance(y_train, pd.DataFrame):
            y_train_arr = y_train.iloc[:, 0].to_numpy()
        else:
            y_train_arr = np.asarray(y_train, dtype=float).ravel()

        forecasts = m4_naive2_forecast(y_train_arr, fh, self.sp)
        return pd.Series(
            forecasts,
            index=fh_index,
            name=getattr(y_true, "name", None),
        )

    def _compute_mase_smape_components(self, y_true, y_pred, y_train, **kwargs):
        """Return aggregate MASE and sMAPE for model and Naive-2 forecasts."""
        y_pred_naive2 = self._predict_naive2(y_train, y_true)
        metric_kwargs = {k: v for k, v in kwargs.items() if k != "y_train"}

        mase = MeanAbsoluteScaledError(
            sp=self.sp,
            multioutput="raw_values",
            multilevel="raw_values",
        )
        smape = MeanAbsolutePercentageError(
            symmetric=True,
            multioutput="raw_values",
            multilevel="raw_values",
        )

        mase_model = mase(y_true, y_pred, y_train=y_train, **metric_kwargs)
        mase_naive2 = mase(y_true, y_pred_naive2, y_train=y_train, **metric_kwargs)
        smape_model = smape(y_true, y_pred, **metric_kwargs)
        smape_naive2 = smape(y_true, y_pred_naive2, **metric_kwargs)

        return mase_model, mase_naive2, smape_model, smape_naive2

    def _owa_from_components(self, mase_model, mase_naive2, smape_model, smape_naive2):
        """Compute OWA from aggregate MASE and sMAPE values."""
        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps

        mase_model = np.asarray(mase_model, dtype=float)
        mase_naive2 = np.asarray(mase_naive2, dtype=float)
        smape_model = np.asarray(smape_model, dtype=float)
        smape_naive2 = np.asarray(smape_naive2, dtype=float)

        mase_ratio = mase_model / np.maximum(mase_naive2, eps)
        smape_ratio = smape_model / np.maximum(smape_naive2, eps)
        owa = np.float64(0.5) * (mase_ratio + smape_ratio)

        if owa.ndim == 0 or owa.size == 1:
            return float(np.ravel(owa)[0])

        if isinstance(owa, pd.DataFrame):
            owa = pd.Series(owa.to_numpy().ravel())
        else:
            owa = pd.Series(np.ravel(owa))

        return owa

    def _compute_owa(self, y_true, y_pred, y_train, **kwargs):
        """Compute aggregate OWA from model and Naive-2 benchmark forecasts."""
        components = self._compute_mase_smape_components(
            y_true, y_pred, y_train, **kwargs
        )
        owa = self._owa_from_components(*components)

        if isinstance(owa, float):
            return owa

        if isinstance(y_true, pd.DataFrame):
            owa.index = y_true.columns[: len(owa)]
        return owa

    def _evaluate(self, y_true, y_pred, **kwargs):
        y_train = kwargs["y_train"]
        metric_kwargs = {k: v for k, v in kwargs.items() if k != "y_train"}
        owa = self._compute_owa(y_true, y_pred, y_train, **metric_kwargs)

        if isinstance(owa, float):
            if self.multioutput == "raw_values":
                columns = y_true.columns if isinstance(y_true, pd.DataFrame) else [0]
                return pd.Series([owa], index=columns[:1])
            return np.float64(owa)

        return self._handle_multioutput(owa, self.multioutput)

    def _evaluate_vectorized(self, y_true, y_pred, **kwargs):
        """Evaluate OWA with M4 competition aggregation across series."""
        if self.multilevel in ["raw_values", "uniform_average_time"]:
            return super()._evaluate_vectorized(y_true, y_pred, **kwargs)

        backend = {}
        backend["backend"] = self.get_config()["backend:parallel"]
        backend["backend_params"] = self.get_config()["backend:parallel:params"]

        component_results = y_true.vectorize_est(
            estimator=self.clone(),
            method="_compute_mase_smape_components",
            varname_of_self="y_true",
            args={**kwargs, "y_pred": y_pred},
            colname_default=self.name,
            return_type="list",
            **backend,
        )

        mase_model_vals = []
        mase_naive2_vals = []
        smape_model_vals = []
        smape_naive2_vals = []

        for components in component_results:
            mase_model, mase_naive2, smape_model, smape_naive2 = components
            mase_model_vals.extend(np.asarray(mase_model, dtype=float).ravel())
            mase_naive2_vals.extend(np.asarray(mase_naive2, dtype=float).ravel())
            smape_model_vals.extend(np.asarray(smape_model, dtype=float).ravel())
            smape_naive2_vals.extend(np.asarray(smape_naive2, dtype=float).ravel())

        owa = self._owa_from_components(
            np.mean(mase_model_vals),
            np.mean(mase_naive2_vals),
            np.mean(smape_model_vals),
            np.mean(smape_naive2_vals),
        )
        return owa

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        y_train = kwargs["y_train"]
        metric_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("y_train", "sample_weight")
        }
        owa = self._compute_owa(y_true, y_pred, y_train, **metric_kwargs)
        owa = np.asarray(owa, dtype=float).ravel()[0]

        owa = pd.DataFrame(
            np.full((len(y_true), y_true.shape[1]), owa),
            index=y_true.index,
            columns=y_true.columns,
        )
        owa = self._get_weighted_df(owa, **kwargs)
        return self._handle_multioutput(owa, self.multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        params1 = {}
        params2 = {"sp": 2}
        return [params1, params2]
