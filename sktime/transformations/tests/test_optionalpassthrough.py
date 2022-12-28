# -*- coding: utf-8 -*-
"""Tests for using OptionalPassthrough."""

import pytest

from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency is not available",
)
def test_optionalpassthrough():
    """Test for OptionalPassthrough used within grid search and with pipeline.

    Same as docstring example of OptionalPassthrough.
    """
    from sklearn.preprocessing import StandardScaler

    from sktime.datasets import load_airline
    from sktime.forecasting.compose import TransformedTargetForecaster
    from sktime.forecasting.model_selection import (
        ForecastingGridSearchCV,
        SlidingWindowSplitter,
    )
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.transformations.compose import OptionalPassthrough
    from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    from sktime.transformations.series.detrend import Deseasonalizer

    # create pipeline
    pipe = TransformedTargetForecaster(
        steps=[
            ("deseasonalizer", OptionalPassthrough(Deseasonalizer())),
            ("scaler", OptionalPassthrough(TabularToSeriesAdaptor(StandardScaler()))),
            ("forecaster", NaiveForecaster()),
        ]
    )
    # putting it all together in a grid search
    cv = SlidingWindowSplitter(
        initial_window=60, window_length=24, start_with_window=True, step_length=48
    )
    param_grid = {
        "deseasonalizer__passthrough": [True, False],
        "scaler__transformer__transformer__with_mean": [True, False],
        "scaler__passthrough": [True, False],
        "forecaster__strategy": ["drift", "mean", "last"],
    }
    gscv = ForecastingGridSearchCV(
        forecaster=pipe, param_grid=param_grid, cv=cv, n_jobs=-1
    )
    gscv.fit(load_airline())
