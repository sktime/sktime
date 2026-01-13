import pandas as pd
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import ForecastingHorizon
import pytest
from sktime.forecasting.compose import (
    DirRecTabularRegressionForecaster,
    DirectTabularRegressionForecaster,
    RecursiveTabularRegressionForecaster,
)
from sktime.utils._testing.hierarchical import _make_hierarchical

@pytest.mark.parametrize("forecaster_cls", [
    DirRecTabularRegressionForecaster,
    DirectTabularRegressionForecaster,
    RecursiveTabularRegressionForecaster,
])
def test_reducer_hierarchical_exogenous_repro(forecaster_cls):
    # Create Hierarchical Data (2 levels, 2 nodes each = 4 instances)
    y = _make_hierarchical(
        hierarchy_levels=(2, 2),
        n_columns=1,
        min_timepoints=20,
        max_timepoints=20,
        index_type="period",
    )
    X = _make_hierarchical(
        hierarchy_levels=(2, 2),
        n_columns=2,  # multivariate exog
        min_timepoints=20,
        max_timepoints=20,
        index_type="period",
    )

    # Split train/test (simple cutoff)
    # We take first 15 points for train, last 5 for test
    # This maintains the structure
    y_train = y.groupby(level=[0, 1]).head(15)
    X_train = X.groupby(level=[0, 1]).head(15)

    # X_test for prediction (future values)
    # We predict 3 steps ahead
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    # X_test must contain the future timepoints for each instance
    # The original X has 20 points. Train has 15 (0..14).
    # Predict 1,2,3 -> indices 15, 16, 17.
    X_test = X.groupby(level=[0, 1]).nth([15, 16, 17])

    forecaster = forecaster_cls(
        estimator=LinearRegression(), window_length=5
    )

    # FIT
    forecaster.fit(y_train, X=X_train, fh=fh)

    # PREDICT
    # Pass X_test covering the forecast horizon
    y_pred = forecaster.predict(fh=fh, X=X_test)

    # Verification
    # 4 instances * 3 timepoints = 12 rows
    assert len(y_pred) == 4 * 3
    assert isinstance(y_pred, pd.DataFrame)
    assert y_pred.index.nlevels == y.index.nlevels
