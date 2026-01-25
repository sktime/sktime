"""Overall Weighted Average (OWA) metric for forecasting."""

import numpy as np

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.param_est.seasonality import SeasonalityACF
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
)
from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
from sktime.transformations.compose import TransformIf
from sktime.transformations.series.detrend import Deseasonalizer

__author__ = ["jgyasu"]


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

    The Naive-2 benchmark is defined as:

    * Detect seasonality using ``SeasonalityACF``
    * If seasonal (sp != 1), apply additive deseasonalization
    * Forecast using ``NaiveForecaster(strategy="last")``

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity of the data.

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
          errors are mean-averaged across levels.
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
    >>> y_true = np.array([100, 110, 105, 120])
    >>> y_pred = np.array([102, 108, 107, 118])
    >>> y_train = np.array([90, 95, 100, 110, 105, 100])
    >>> metric = OverallWeightedAverage(sp=1)
    >>> metric(y_true, y_pred, y_train=y_train)
    np.float64(0.14512226890252225)
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

    def _make_naive2_forecaster(self):
        """Construct the Naive-2 benchmark forecaster."""
        return TransformIf(
            SeasonalityACF(candidate_sp=self.sp),
            "sp",
            "!=",
            1,
            Deseasonalizer(sp=self.sp, model="additive"),
        ) * NaiveForecaster(strategy="last")

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        y_train = kwargs["y_train"]
        index = y_true.index

        if hasattr(index, "nlevels") and index.nlevels > 1:
            fh_index = index.get_level_values(-1).unique()
        else:
            fh_index = index

        fh = ForecastingHorizon(fh_index, is_relative=False)

        naive2 = self._make_naive2_forecaster()
        naive2.fit(y_train, fh=fh)
        y_pred_naive2 = naive2.predict(fh)

        mase = MeanAbsoluteScaledError(
            sp=self.sp,
            multioutput="raw_values",
            multilevel="raw_values",
            by_index=True,
        )
        smape = MeanAbsolutePercentageError(
            symmetric=True,
            multioutput="raw_values",
            multilevel="raw_values",
            by_index=True,
        )

        mase_model = mase(y_true, y_pred, y_train=y_train)
        mase_naive2 = mase(y_true, y_pred_naive2, y_train=y_train)

        smape_model = smape(y_true, y_pred)
        smape_naive2 = smape(y_true, y_pred_naive2)

        eps = self.eps
        if eps is None:
            eps = np.finfo(np.float64).eps
        mase_ratio = mase_model / np.maximum(mase_naive2, eps)
        smape_ratio = smape_model / np.maximum(smape_naive2, eps)

        owa = np.float64(0.5) * (mase_ratio + smape_ratio)

        owa = self._get_weighted_df(owa, **kwargs)
        return self._handle_multioutput(owa, self.multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        params1 = {}
        params2 = {"sp": 2}
        return [params1, params2]
