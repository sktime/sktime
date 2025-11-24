import numpy as np

from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
)
from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class OWA(BaseForecastingErrorMetric):
    """Overall Weighted Average (OWA) used in the M4 competition.

    OWA = 0.5 * ( MASE / MASE_Naive2  +  sMAPE / sMAPE_Naive2 )

    Parameters
    ----------
    sp : int, default=1
        Seasonal periodicity for computing MASE.
    """

    def __init__(self, sp=1):
        self.sp = sp
        super().__init__()

        # internal metrics
        self._mase = MeanAbsoluteScaledError(sp=sp)
        self._smape = MeanAbsolutePercentageError(symmetric=True)

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Compute overall weighted average (OWA).

        Required kwarg:
        y_benchmark : pd.DataFrame or array
            The Naive2 benchmark forecast.
        y_train : pd.DataFrame or array
            Training data (required for MASE denominator)
        """
        if "y_benchmark" not in kwargs:
            raise ValueError("OWA requires `y_benchmark` argument (Naive2 forecast).")

        y_benchmark = kwargs["y_benchmark"]
        y_train = kwargs.get("y_train", None)

        # method errors
        mase = self._mase(y_true, y_pred, y_train=y_train)
        smape = self._smape(y_true, y_pred)

        # benchmark errors (Naive2)
        mase_bench = self._mase(y_true, y_benchmark, y_train=y_train)
        smape_bench = self._smape(y_true, y_benchmark)

        # compute OWA
        owa = 0.5 * ((mase / mase_bench) + (smape / smape_bench))
        return np.float64(owa)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return {"sp": 1}
