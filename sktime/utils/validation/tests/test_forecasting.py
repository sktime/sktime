#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test forecasting module."""

__author__ = ["mloning"]

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from sktime.forecasting.var import VAR
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.plotting import plot_series
from sktime.utils.validation.forecasting import check_fh

empty_input = (np.array([], dtype=int), [], pd.Index([], dtype=int))


@pytest.mark.skipif(
    not run_test_for_class(check_fh),
    reason="Run if tested function has changed.",
)
@pytest.mark.parametrize("arg", empty_input)
def test_check_fh_empty_input(arg):
    """Test that fh validation throws an error with empty container."""
    with raises(ValueError, match="`fh` must not be empty"):
        check_fh(arg)


def test_plotting_dataframe_with_unused_levels():
    """After subsetting, pd.DataFrame keeps information about unused levels."""

    _check_soft_dependencies("matplotlib")

    import matplotlib
    import matplotlib.pyplot as plt

    y_train = pd.DataFrame(np.random.rand(100, 3), columns=["x1", "x2", "x3"])
    forecaster = VAR(maxlags=1)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=range(1, 10))
    coverage = 0.9
    y_pred_ints1 = forecaster.predict_interval(coverage=coverage)
    variable_to_plot = "x1"
    pred_interval = y_pred_ints1[[variable_to_plot]].copy()

    matplotlib.use("agg")
    plot_series(
        y_train[[variable_to_plot]],
        y_pred[[variable_to_plot]],
        pred_interval=pred_interval,
    )
    plt.gcf().canvas.draw_idle()
    plt.close()
