"""Regression tests for bugfixes related to base class related functionality."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pytest
from skbase.utils.dependencies import _check_estimator_deps

from sktime.forecasting.compose import ForecastByLevel, TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.difference import Differencer
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _make_hierarchical


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
def test_statsmodels_adapter_random_state_handling():
    """Regression test for #10968: avoid passing unsupported random_state."""
    import pandas as pd

    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.base.adapters._statsmodels import _StatsModelsAdapter

    class MockPredictionResults:
        def conf_int(self, alpha):
            return pd.DataFrame([[0, 1], [0, 1], [0, 1]], columns=["lower", "upper"])

    class MockNonETSModel:
        def get_prediction(self, start=None, end=None, **kwargs):
            assert "random_state" not in kwargs
            return MockPredictionResults()

    class MockETSModel:
        def get_prediction(
            self,
            start=None,
            end=None,
            dynamic=False,
            index=None,
            method=None,
            simulate_repetitions=1000,
            **simulate_kwargs,
        ):
            assert simulate_kwargs["simulate_kwargs"] == {"rng": 42}
            return MockPredictionResults()

        def simulate(self, nsimulations, rng=None, **kwargs):
            return None

    class MockAdapter(_StatsModelsAdapter):
        _tags = {
            "capability:pred_int": True,
        }

        def __init__(self, model, random_state=None):
            self.model = model
            super().__init__(random_state=random_state)

        def _fit_forecaster(self, y, X=None):
            self._fitted_forecaster = self.model

        @staticmethod
        def _extract_conf_int(prediction_results, alpha):
            return prediction_results.conf_int(alpha)

    y = pd.Series([1, 2, 3, 4, 5])
    fh = ForecastingHorizon([1, 2, 3])

    non_ets = MockAdapter(MockNonETSModel(), random_state=42)
    non_ets.fit(y, fh=fh)
    non_ets.predict_interval(fh=fh, coverage=[0.9])

    ets = MockAdapter(MockETSModel(), random_state=42)
    ets.fit(y, fh=fh)
    ets.predict_interval(fh=fh, coverage=[0.9])
