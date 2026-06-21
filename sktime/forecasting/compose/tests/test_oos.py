#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for InsampleForecaster."""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.compose import InsampleForecaster, make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter, temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(InsampleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_oos_forecaster_all_in_sample():
    """Test all in-sample predictions."""
    y = load_airline()

    initial_window = 12

    forecaster = make_reduction(
        estimator=LinearRegression(),
        strategy="recursive",
        window_length=initial_window - 1,
    )
    cv = ExpandingWindowSplitter(initial_window=initial_window)
    wrapper = InsampleForecaster(
        forecaster=forecaster,
        cv=cv,
        strategy="refit",
    )

    assert not forecaster.get_tag("capability:insample")
    assert wrapper.get_tag("capability:insample")

    wrapper.fit(y, fh=y.index)
    preds = wrapper.predict()

    assert preds.iloc[:initial_window].isna().all()
    assert preds.iloc[initial_window:].notna().all()
    pd.testing.assert_index_equal(preds.index, y.index)


@pytest.mark.skipif(
    not run_test_for_class(InsampleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_oos_forecaster_all_out_of_sample():
    """Test all out-of-sample predictions."""
    y = load_airline()

    fh = [1, 2, 3, 4]

    forecaster = make_reduction(
        estimator=LinearRegression(),
        strategy="recursive",
        window_length=12,
    )
    wrapper = InsampleForecaster(forecaster=forecaster)

    forecaster.fit(y, fh=fh)
    forecaster_preds = forecaster.predict()

    wrapper.fit(y, fh=fh)
    wrapper_preds = wrapper.predict()

    pd.testing.assert_series_equal(
        forecaster_preds,
        wrapper_preds,
        check_names=False,  # original forecaster may not set name
    )


@pytest.mark.skipif(
    not run_test_for_class(InsampleForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "forecaster",
    [
        NaiveForecaster(strategy="drift"),
        make_reduction(
            estimator=LinearRegression(),
            strategy="recursive",
            window_length=2,
        ),
    ],
)
@pytest.mark.parametrize("strategy", ["refit", "update", "no-update_params"])
@pytest.mark.parametrize(
    "fh",
    [
        [1, 2, 3, 4],
        [-3, -2, -1, 0],
        [-3, -1, 0, 1, 3],
    ],
)
def test_oos_forecaster_mixed_config(forecaster, strategy, fh):
    """Test on different combinations."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)

    cv = ExpandingWindowSplitter(initial_window=3)
    wrapper = InsampleForecaster(
        forecaster=forecaster,
        cv=cv,
        strategy=strategy,
        backend="loky",
    )

    wrapper.fit(y=y_train, X=X_train, fh=fh)
    preds = wrapper.predict(fh, X=X_test)

    fh_corrected = [y_train.shape[0] + i - 1 for i in fh]
    y_corrected = y.iloc[fh_corrected]
    pd.testing.assert_index_equal(preds.index, y_corrected.index)
