# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import clone

from sktime.annotation.base._base import BaseSeriesAnnotator
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

__author__ = ["satya-pattnaik", "mloning"]


def check_estimator(estimator):
    """
    Check validity of estimator for `QuantileOutlierDetector`.

    Check if the `estimator` object passed as the argument is a valid Sktime
    Forecaster and if the `estimator` generates a prediction/confidence interval.

    Parameters
    ----------
    estimator : Sktime Forecaster
    Raises
    -------
    ValueError: if estimator is not a valid forecaster and
    does not generates a prediction/confidence interval
    """
    if not isinstance(estimator, BaseForecaster):
        raise ValueError("estimator must be a forecaster")
    if not estimator.get_tag("capability:pred_int", tag_value_default=False):
        raise ValueError(
            f"{estimator.__class__.__name__} does not support prediction quantiles."
        )


class QuantileOutlierDetector(BaseSeriesAnnotator):
    r"""An Outlier Detector based on Confidence/Prediction Intervals.

    This method is inspired from : [1]

    An Outlier Detector which detects those time points as outliers for
    which the values fall outside of the Uncertainty Intervals of forecasts
    generated from a Sktime Forecaster.

    [1]https://docs.seldon.io/projects/alibi-detect/en/latest/methods/prophet.html

    Parameters
    ----------
    estimator : a Sktime Forecaster object
    fmt : str {"dense", "sparse"}, optional (default="dense")
        Annotation output format:
        * If "sparse", a sub-series of labels for only the outliers in X is returned,
        * If "dense", a series of labels for all values in X is returned.
    labels : str {"indicator", "score"}, optional (default="indicator")
        Annotation output labels:
        * If "indicator", returned values are boolean, indicating whether a value is an
        outlier,
        * If "score", returned values are floats, giving the outlier score.
    alpha : float, Significance level of the Uncertainty Interval to be used,
            for detecting the Outliers,(default=0.05)
    """

    def __init__(self, estimator, fmt="dense", labels="indicator", alpha=0.05):
        self.estimator = estimator  # Sktime forecaster
        self.alpha = alpha
        super(QuantileOutlierDetector, self).__init__(fmt=fmt, labels=labels)

    def _fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised
        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        
        self.estimator_ = clone(self.estimator)
        check_estimator(self.estimator_)
        self.estimator_.fit(X)
        return self

    def _predict(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        
        fmt = self.fmt
        labels = self.labels

        fh = ForecastingHorizon(X.index, is_relative=False)
        forecasts, uncertainty_intervals = self.estimator_.predict(
            fh, return_pred_int=True, alpha=self.alpha
        )

        uncertainty_intervals["forecast"] = forecasts
        uncertainty_intervals["original"] = X.to_numpy()

        uncertainty_intervals["score"] = (
            uncertainty_intervals["original"] - uncertainty_intervals["upper"]
        ) * (uncertainty_intervals["original"] >= uncertainty_intervals["forecast"]) + (
            uncertainty_intervals["lower"] - uncertainty_intervals["original"]
        ) * (
            uncertainty_intervals["original"] < uncertainty_intervals["forecast"]
        )
        uncertainty_intervals["is_outlier"] = (
            uncertainty_intervals["score"] > 0.0
        ).astype(int)
        if labels == "score":
            Y_val_np = uncertainty_intervals["score"]
        elif labels == "indicator":
            Y_val_np = uncertainty_intervals["is_outlier"]

        if fmt == "dense":
            Y = pd.Series(Y_val_np, index=X.index)
        elif fmt == "sparse":
            Y_loc = np.where(Y_val_np)
            Y = pd.Series(Y_val_np.iloc[Y_loc], index=X.index[Y_loc])

        return Y

    @classmethod
    def get_test_params(cls):
        from sktime.forecasting.arima import ARIMA
        return {"estimator":ARIMA()}