# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests the conformal interval wrapper."""

__author__ = ["fkiraly", "bethrice44"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.datatypes import convert_to, scitype_to_mtype
from sktime.forecasting.conformal import ConformalIntervals
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.naive import NaiveForecaster, NaiveVariance
from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
from sktime.tests.test_switch import run_test_for_class

INTERVAL_WRAPPERS = [ConformalIntervals, NaiveVariance]
CV_SPLITTERS = [SlidingWindowSplitter, ExpandingWindowSplitter]
EVALUATE_STRATEGY = ["update", "refit"]
SAMPLE_FRACS = [None, 0.5]
MTYPES_SERIES = scitype_to_mtype("Series", softdeps="present")


@pytest.mark.skipif(
    not run_test_for_class(INTERVAL_WRAPPERS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("mtype", MTYPES_SERIES)
@pytest.mark.parametrize("override_y_mtype", [True, False])
@pytest.mark.parametrize("wrapper", INTERVAL_WRAPPERS)
def test_wrapper_series_mtype(wrapper, override_y_mtype, mtype):
    """Test that interval wrappers behave nicely with different internal y_mtypes.

    The wrappers require y to be pd.Series, and the internal estimator can have
    a different internal mtype.

    We test all interval wrappers in sktime (wrapper).

    We test once with an internal forecaster that needs pd.DataFrame conversion,
    and one that accepts pd.Series.
    We do this with a trick: the vanilla NaiveForecaster can accept both; we mimic a
    "pd.DataFrame only" forecaster by restricting its y_inner_mtype tag to pd.Series.
    """
    y = load_airline()
    y = convert_to(y, to_type=mtype)

    f = NaiveForecaster()

    if override_y_mtype:
        f.set_tags(**{"y_inner_mtype": "pd.DataFrame"})

    interval_forecaster = wrapper(f)
    interval_forecaster.fit(y, fh=[1, 2, 3])
    pred_int = interval_forecaster.predict_interval()

    assert isinstance(pred_int, pd.DataFrame)
    assert len(pred_int) == 3

    pred_var = interval_forecaster.predict_var()

    assert isinstance(pred_var, pd.DataFrame)
    assert len(pred_var) == 3


@pytest.mark.skipif(
    not run_test_for_class(INTERVAL_WRAPPERS + [evaluate]),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("wrapper", INTERVAL_WRAPPERS)
@pytest.mark.parametrize("splitter", CV_SPLITTERS)
@pytest.mark.parametrize("strategy", EVALUATE_STRATEGY)
@pytest.mark.parametrize("sample_frac", SAMPLE_FRACS)
def test_evaluate_with_window_splitters(wrapper, splitter, strategy, sample_frac):
    """Test interval wrappers with different strategies and cross validators.

    The wrapper does some internal sliding window cross-validation to calculate the
    `residuals_matrix`, which means the initial cross-validation can cause issues.

    This checks refit and update strategies as well as expanding and sliding window
    splitters.
    """
    y = load_airline()[:60]

    if splitter == SlidingWindowSplitter:
        cv = splitter(
            fh=np.arange(1, 7),
            window_length=24,
            step_length=6,
        )
    elif splitter == ExpandingWindowSplitter:
        cv = splitter(
            fh=np.arange(1, 7),
            initial_window=24,
            step_length=6,
        )

    f = NaiveForecaster()

    if wrapper == ConformalIntervals:
        interval_forecaster = wrapper(f, initial_window=12, sample_frac=sample_frac)
    else:
        interval_forecaster = wrapper(f, initial_window=12)

    results = evaluate(
        forecaster=interval_forecaster,
        cv=cv,
        y=y,
        X=None,
        strategy=strategy,
        scoring=PinballLoss(alpha=[0.1, 0.5, 0.9]),
        return_data=True,
        error_score="raise",
        backend=None,
    )

    assert len(results) == 6
    assert not results.test_PinballLoss.isna().any()
