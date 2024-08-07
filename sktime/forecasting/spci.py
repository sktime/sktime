# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements SPCI Forecaster."""

_all_ = ["SPCIForecaster"]
__author__ = ["ksharma6"]


from sktime.forecasting.base import BaseForecaster

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class SPCI(BaseForecaster):
    """Sequential Predictive Conformal Inference Forecaster.

    SPCI is a model-free and distribution-free framework that combines
    a Sktime forecaster with a quantile regression model to perform Conformal
    Predictions on time series data [1]

    The algorithm works like so:
    1. Obtain point predictions, ``y_preds``, and point prediction residuals,
    ``e^= y - y_preds``, using Sktime forecaster.
    2. For t > T do`:
        3. Fit quantile regressor onto ``e^``
        4. Use quantile regression to obtain quantile predictions, ``q_pred``
        5. Calculate prediction interval at time ``t``
        6. Calculate the new residual ``e^_t``
        7. Update residuals ``e^`` by sliding one index forward
        (i.e. add ``e^_t`` and remove the oldest one)

    returns: Prediction intervals

    Parameters (need to update)
    ----------
    forecaster : estimator
        The base forecaster for point predictions.
    regressor: quantile regression model
        Regressor predicts
    alpha: float, significance level, default=.05
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.


    Examples
    --------
    continue

    References
    ----------
    .. [1] Chen Xu & Yao Xie (2023). Sequential Predictive
    Conformal Inference for Time Series.

    """

    pass
