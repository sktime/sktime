"""Tests for checking with categorical inputs

Cases where error must be raised are tested in test_all_forecasters in
- test_categorical_y_raises_error
- test_categorical_X_raises_error
"""

__author__ = ["Abhay-Lejith"]


import pandas as pd

from sktime.forecasting.dummy import ForecastKnownValues


def test_est_with_categorical_capability():
    """Test that categorical data works when native support is available.

    This test uses the dummy forecaster with modified tags to imitate a forecaster
    which supports categorical natively in exogeneous X for checking whether
    categorical data passes through the boilerplate checks without error.
    """
    y = pd.DataFrame(range(9))

    est = ForecastKnownValues(y)
    modified_tags = {
        "ignores-exogeneous-X": False,
        "capability:categorical_in_X": True,
    }
    est.set_tags(**modified_tags)

    yt = y[:6]
    X = pd.DataFrame({"col_0": ["a", "b", "c", "a", "b", "c", "a", "b", "c"]})
    Xt = X[:6]

    est.fit(yt, Xt, fh=[1, 2, 3])
    est.predict(X=X[6:])
