"""Regression tests for bugfixes related to base class related functionality."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import ForecastByLevel, TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.transformations.series.difference import Differencer
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.dependencies import _check_estimator_deps


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.base")
    or not _check_estimator_deps(ExponentialSmoothing, severity="none"),
    reason="run only if base module has changed",
)
def test_heterogeneous_get_fitted_params():
    """Regression test for bugfix #4574, related to get_fitted_params."""
    y = _make_hierarchical(hierarchy_levels=(2, 2), min_timepoints=7, max_timepoints=7)
    agg = Aggregator()
    y_agg = agg.fit_transform(y)

    param_grid = [
        {
            "forecaster": [ExponentialSmoothing()],
            "forecaster__trend": ["add", "mul"],
        },
        {
            "forecaster": [PolynomialTrendForecaster()],
            "forecaster__degree": [1, 2],
        },
    ]

    pipe = TransformedTargetForecaster(steps=[("forecaster", ExponentialSmoothing())])

    N_cv_fold = 2
    step_cv = 1
    fh = [1, 2]

    N_t = len(y_agg.index.get_level_values(2).unique())
    initial_window_cv_len = N_t - (N_cv_fold - 1) * step_cv - fh[-1]

    cv = ExpandingWindowSplitter(
        initial_window=initial_window_cv_len,
        step_length=step_cv,
        fh=fh,
    )

    gscv = ForecastingGridSearchCV(forecaster=pipe, param_grid=param_grid, cv=cv)
    gscv_bylevel = ForecastByLevel(gscv, "local")
    reconciler = ReconcilerForecaster(gscv_bylevel, method="ols")

    reconciler.fit(y_agg)
    reconciler.get_fitted_params()  # triggers an error pre-fix


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.base"),
    reason="run only if base module has changed",
)
def test_predict_residuals_conversion():
    """Regression test for bugfix #4766, related to predict_residuals internal type."""
    from sktime.datasets import load_longley
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = Differencer() * NaiveForecaster()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict_residuals()

    assert type(result) is type(y_train)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.base"),
    reason="run only if base module has changed",
)
def test_update_predict_with_x():
    """Regression test for bugfix #6026, related to update_predict windowing."""
    from sklearn.linear_model import Lasso

    from sktime.forecasting.compose import make_reduction
    from sktime.split import temporal_train_test_split

    # Set seed for reproducibility
    np.random.seed(0)

    # Generate synthetic data
    num_rows = 300
    data = {
        "open": np.random.uniform(10000, 20000, num_rows),
        "high": np.random.uniform(10000, 20000, num_rows),
        "low": np.random.uniform(10000, 20000, num_rows),
        "close": np.random.uniform(10000, 20000, num_rows),
        "target": np.random.uniform(10000, 20000, num_rows),
    }

    # Create DataFrame
    synthetic_df = pd.DataFrame(data)

    # Set index to DatetimeIndex similar to the provided index
    synthetic_df.index = pd.date_range(start="2020-08-06", periods=num_rows, freq="D")

    y = synthetic_df[["target"]]
    X = synthetic_df[["open", "high", "low", "close"]]

    test_size = int(len(X) * 0.33)
    fh = ForecastingHorizon(np.arange(1, 3), is_relative=True)

    y_train, y_test, X_train, X_test = temporal_train_test_split(
        y, X, test_size=test_size
    )

    pipe = make_reduction(Lasso(alpha=0.1))

    fh = ForecastingHorizon(np.arange(1, 2), is_relative=True)

    # Fit the model
    pipe.fit(y=y_train, X=X_train, fh=fh)

    # Make forecasts
    pipe.update_predict(y=y_test, X=X_test)
