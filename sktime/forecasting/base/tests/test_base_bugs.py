"""Regression tests for bugfixes related to base class related functionality."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pytest

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
def test_base_global_forecaster_rejects_y_without_capability():
    """Regression test for bugfix #10200.

    ``_BaseGlobalForecaster`` constructed a ``ValueError("no global forecasting
    support!")`` without ``raise`` in ``predict``, ``predict_quantiles``,
    ``predict_interval``, ``predict_var`` and ``predict_proba``. Passing ``y`` to
    a forecaster lacking global forecasting capability therefore silently passed
    through instead of erroring. All five methods must now raise ``ValueError``.
    """
    import pandas as pd

    from sktime.forecasting.base._deprecation_global import _BaseGlobalForecaster

    class _NonGlobalForecaster(_BaseGlobalForecaster):
        # capability:global_forecasting is left at its default of False;
        # capability:pred_int is enabled so the probabilistic predict methods
        # reach the y-rejection check rather than their own capability guard.
        _tags = {"capability:pred_int": True}

        def _fit(self, y, X, fh):
            return self

        def _predict(self, fh, X=None):
            return pd.Series([0.0])

    forecaster = _NonGlobalForecaster()
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    for method in [
        "predict",
        "predict_quantiles",
        "predict_interval",
        "predict_var",
        "predict_proba",
    ]:
        with pytest.raises(ValueError, match="no global forecasting support"):
            getattr(forecaster, method)(fh=[1], y=y)
