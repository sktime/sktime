"""Regression tests for bugfixes related to base class related functionality."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_estimator_deps, _check_soft_dependencies

from sktime.forecasting.compose import ForecastByLevel, TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.reconcile import ReconcilerForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.split import ExpandingWindowSplitter
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.hierarchical import _make_hierarchical


@pytest.mark.parametrize("n_columns", [1, 2])
def test_update_predict_with_numpy_series(n_columns):
    """Regression test for #3291: update_predict supports NumPy input."""
    y = np.arange(10, dtype=float)
    if n_columns == 2:
        y = np.column_stack([y, y + 100])
    forecaster = NaiveForecaster(strategy="last").fit(y[:5])

    y_pred = forecaster.update_predict(y[5:])

    assert isinstance(y_pred, np.ndarray)
    expected = y[5:-1, None] if n_columns == 1 else y[5:-1]
    np.testing.assert_array_equal(y_pred, expected)


@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("remember_data", [True, False])
@pytest.mark.parametrize("reset_forecaster", [True, False])
def test_update_predict_preserves_numpy_output_state(
    n_columns, remember_data, reset_forecaster
):
    """Internal pandas windows must not replace the public NumPy output type."""
    y = np.arange(14, dtype=float)
    if n_columns == 2:
        y = np.column_stack([y, y + 100])
    original_y = y.copy()
    forecaster = (
        NaiveForecaster(strategy="last")
        .set_config(remember_data=remember_data)
        .fit(y[:5])
    )

    y_pred = forecaster.update_predict(y[5:10], reset_forecaster=reset_forecaster)

    assert isinstance(y_pred, np.ndarray)
    expected = y[5:9, None] if n_columns == 1 else y[5:9]
    np.testing.assert_array_equal(y_pred, expected)
    assert forecaster.cutoff[0] == (4 if reset_forecaster else 8)

    if not reset_forecaster:
        y_pred = forecaster.update_predict(y[9:14], reset_forecaster=False)
        assert isinstance(y_pred, np.ndarray)
        expected = y[9:13, None] if n_columns == 1 else y[9:13]
        np.testing.assert_array_equal(y_pred, expected)
        assert forecaster.cutoff[0] == 12

    next_pred = forecaster.predict(fh=[1])
    assert isinstance(next_pred, np.ndarray)
    last = 4 if reset_forecaster else 12
    expected = y[last : last + 1, None] if n_columns == 1 else y[last : last + 1]
    np.testing.assert_array_equal(next_pred, expected)
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("reset_forecaster", [True, False])
def test_update_predict_preserves_pandas_output_state(n_columns, reset_forecaster):
    """Preserve public names and isolate vectorized models for pandas too."""
    y = pd.Series(np.arange(10, dtype=float), name="target")
    if n_columns == 2:
        y = pd.DataFrame({"target": y, "other": y + 100})
    forecaster = NaiveForecaster(strategy="last").fit(y.iloc[:5])

    y_pred = forecaster.update_predict(y.iloc[5:], reset_forecaster=reset_forecaster)
    expected = y.iloc[5:9].copy()
    expected.index = pd.RangeIndex(6, 10)
    assert_equal = pd.testing.assert_series_equal
    if n_columns == 2:
        assert_equal = pd.testing.assert_frame_equal
    assert_equal(y_pred, expected)

    last = 4 if reset_forecaster else 8
    expected = y.iloc[last : last + 1].copy()
    expected.index = pd.RangeIndex(last + 1, last + 2)
    assert_equal(forecaster.predict(fh=[1]), expected)


@pytest.mark.parametrize("n_columns", [1, 2])
def test_update_predict_preserves_numpy_type_after_error(n_columns, monkeypatch):
    """A failed rolling update must not leak internal pandas output metadata."""
    y = np.arange(10, dtype=float)
    if n_columns == 2:
        y = np.column_stack([y, y + 100])
    forecaster = NaiveForecaster(strategy="last").fit(y[:5])
    original_update_predict_single = NaiveForecaster.update_predict_single

    def update_then_fail(self, *args, **kwargs):
        prediction = original_update_predict_single(self, *args, **kwargs)
        if self is forecaster:
            raise RuntimeError("rolling update failed")
        return prediction

    monkeypatch.setattr(NaiveForecaster, "update_predict_single", update_then_fail)

    with pytest.raises(RuntimeError, match="rolling update failed"):
        forecaster.update_predict(y[5:], reset_forecaster=False)

    y_pred = forecaster.predict(fh=[1])
    assert isinstance(y_pred, np.ndarray)
    expected = y[5:6, None] if n_columns == 1 else y[5:6]
    np.testing.assert_array_equal(y_pred, expected)
    assert forecaster.cutoff[0] == 5


def test_update_predict_with_numpy_series_and_multiple_horizons():
    """NumPy rolling predictions preserve the fitted series time axis."""
    y = np.arange(10, dtype=float)
    cv = ExpandingWindowSplitter(fh=[1, 2], initial_window=2, step_length=1)

    numpy_forecaster = NaiveForecaster(strategy="last").fit(y[:5])
    numpy_pred = numpy_forecaster.update_predict(y[5:], cv=cv, update_params=False)

    pandas_y = pd.Series(y)
    pandas_forecaster = NaiveForecaster(strategy="last").fit(pandas_y.iloc[:5])
    pandas_pred = pandas_forecaster.update_predict(
        pandas_y.iloc[5:], cv=cv, update_params=False
    )

    pd.testing.assert_frame_equal(numpy_pred, pandas_pred)


def test_update_predict_with_numpy_exogenous_series():
    """NumPy exogenous data uses the same continuation index as NumPy y."""
    y = np.arange(10, dtype=float)
    X = np.arange(20, dtype=float).reshape(10, 2)
    forecaster = NaiveForecaster(strategy="last").fit(y[:5], X=X[:5])

    y_pred = forecaster.update_predict(y[5:], X=X[5:], update_params=False)

    np.testing.assert_array_equal(y_pred, y[5:-1, None])


def test_update_predict_with_numpy_after_datetime_series():
    """NumPy updates continue a fitted datetime index and return NumPy output."""
    y_train = pd.Series(
        np.arange(5, dtype=float),
        index=pd.date_range("2026-01-01", periods=5, freq="D"),
    )
    forecaster = NaiveForecaster(strategy="last").fit(y_train)

    y_pred = forecaster.update_predict(np.arange(5, 10, dtype=float))

    assert isinstance(y_pred, np.ndarray)
    np.testing.assert_array_equal(y_pred, np.arange(5, 9, dtype=float)[:, None])


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.base")
    or not _check_estimator_deps(ExponentialSmoothing, severity="none"),
    reason="run only if base module has changed",
)
def test_heterogeneous_get_fitted_params():
    """Regression test for bugfix #4574, related to get_fitted_params."""
    from sktime.transformations.hierarchical.aggregate import Aggregator

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
    from sktime.transformations.difference import Differencer

    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    pipe = Differencer() * NaiveForecaster()
    pipe.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    result = pipe.predict_residuals(y_train)

    assert type(result) is type(y_train)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.forecasting.base")
    or not _check_soft_dependencies("statsmodels", severity="none"),
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
