"""Overall Weighted Average (OWA) metric for forecasting."""

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
    """Overall Weighted Average (OWA) as used in the M4 competition.

    The Overall Weighted Average (OWA) metric combines a scale-dependent error
    (Mean Absolute Scaled Error, MASE) and a scale-independent error
    (symmetric Mean Absolute Percentage Error, sMAPE) into a single normalized
    score. The normalization is performed relative to a fixed benchmark
    forecast, Naive2, following the official definition from the M4 competition.

    The metric is defined as:

        OWA = 0.5 * ( MASE / MASE_Naive2  +  sMAPE / sMAPE_Naive2 )

    where `MASE_Naive2` and `sMAPE_Naive2` are the corresponding errors
    obtained from the Naive2 benchmark forecaster.

    In this implementation, the Naive2 benchmark is constructed as follows:

    * Seasonality is detected from the training series using
        ``SeasonalityACF`` with ``candidate_sp=sp``.
    * If a seasonal period other than 1 is detected, the series is
       deseasonalized using an additive ``Deseasonalizer`` with period ``sp``.
    * A ``NaiveForecaster(strategy="last")`` is fitted to the (optionally)
       deseasonalized series.
    * Forecasts are produced for the required forecasting horizon and
       reseasonalized if deseasonalization was applied.

    An OWA score of 1 indicates performance equal to the Naive2 benchmark.
    Scores below 1 indicate better performance, while scores above 1 indicate
    worse performance.

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity of the time series. This is used both for computing
        the MASE denominator and for constructing the Naive2 benchmark via
        seasonal deseasonalization.

    References
    ----------
    Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020).
    The M4 Competition: 100,000 time series and 61 forecasting methods.
    International Journal of Forecasting, 36(1), 54-74.
    https://doi.org/10.1016/j.ijforecast.2019.04.014

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import OverallWeightedAverage
    >>> y_true = np.array([100, 110, 105, 120])
    >>> y_pred = np.array([102, 108, 107, 118])
    >>> y_train = np.array([90, 95, 100, 110, 105, 120])
    >>> metric = OverallWeightedAverage(sp=12)
    >>> metric(y_true, y_pred, y_train=y_train)
    np.float64(0.12954147316682615)
    """

    _tags = {
        "requires-y-train": True,
        "python_dependencies": ["statsmodels"],
    }

    def __init__(self, sp=1):
        self.sp = sp
        super().__init__()

        self._mase = MeanAbsoluteScaledError(sp=sp)
        self._smape = MeanAbsolutePercentageError(symmetric=True)

        self._naive2 = TransformIf(
            SeasonalityACF(candidate_sp=self.sp),
            "sp",
            "!=",
            1,
            Deseasonalizer(sp=self.sp, model="additive"),
        ) * NaiveForecaster(strategy="last")

    def _evaluate(self, y_true, y_pred, **kwargs):
        y_train = kwargs["y_train"]

        fh = self._get_fh_from_y_pred(y_pred)

        naive2 = self._naive2.clone()
        naive2.fit(y_train)
        y_pred_benchmark = naive2.predict(fh)

        mase = self._mase(y_true, y_pred, y_train=y_train)
        smape = self._smape(y_true, y_pred)

        mase_bench = self._mase(y_true, y_pred_benchmark, y_train=y_train)
        smape_bench = self._smape(y_true, y_pred_benchmark)

        return 0.5 * ((mase / mase_bench) + (smape / smape_bench))

    def _get_fh_from_y_pred(self, y_pred):
        return ForecastingHorizon(y_pred.index, is_relative=False)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"sp": 1}
        params2 = {"sp": 12}
        return [params1, params2]
