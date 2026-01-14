from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
)
from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric

__author__ = ["jgyasu"]


class OverallWeightedAverage(BaseForecastingErrorMetric):
    """Overall Weighted Average (OWA) as used in the M4 competition.

    The OWA metric combines scale-dependent (MASE) and scale-independent
    (sMAPE) errors into a single normalized score relative to a benchmark
    forecast, following the official definition from the M4 forecasting
    competition.

    The metric is defined as:

        OWA = 0.5 * ( MASE / MASE_Naive2  +  sMAPE / sMAPE_Naive2 )

    The OWA score equals 1 when the evaluated method performs exactly as
    well as the Naive2 benchmark. Scores below 1 indicate better performance;
    scores above 1 indicate worse performance.

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity for computing MASE.

    References
    ----------
    Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020).
    The M4 Competition: 100,000 time series and 61 forecasting methods.
    International Journal of Forecasting, 36(1), 54-74.
    https://doi.org/10.1016/j.ijforecast.2019.04.014

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.performance_metrics.forecasting import OverallWeightedAverage
    >>> y_true = np.array([100, 110, 105, 120])
    >>> y_pred = np.array([102, 108, 107, 118])
    >>> y_train = np.array([90, 95, 100, 110, 105, 120])
    >>> y_pred_benchmark = np.array([105, 100, 110, 105])
    >>> metric = OverallWeightedAverage(sp=2)
    >>> metric(y_true, y_pred, y_pred_benchmark=y_pred_benchmark, y_train=y_train)
    np.float64(0.22826157321109153)
    """

    _tags = {
        "requires-y-train": True,
        "requires-y-pred-benchmark": True,
    }

    def __init__(self, sp=1):
        self.sp = sp
        super().__init__()

        self._mase = MeanAbsoluteScaledError(sp=sp)
        self._smape = MeanAbsolutePercentageError(symmetric=True)

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Compute overall weighted average (OWA).

        Required kwarg:
        y_pred_benchmark : pd.DataFrame or array
            The Naive2 benchmark forecast.
        y_train : pd.DataFrame or array
            Training data (required for MASE denominator)
        """
        y_pred_benchmark = kwargs["y_pred_benchmark"]
        y_train = kwargs.get("y_train", None)

        mase = self._mase(y_true, y_pred, y_train=y_train)
        smape = self._smape(y_true, y_pred)

        mase_bench = self._mase(y_true, y_pred_benchmark, y_train=y_train)
        smape_bench = self._smape(y_true, y_pred_benchmark)

        owa = 0.5 * ((mase / mase_bench) + (smape / smape_bench))
        return owa

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
        params1 = {}
        params2 = {"sp": None}
        return [params1, params2]
