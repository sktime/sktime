# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for IgnoreX."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from sktime.datasets import load_longley
from sktime.forecasting.compose import IgnoreX
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(IgnoreX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("ignore_x", [True, False, None])
def test_ignoreX(ignore_x):
    """Test that indeed X=None is passed iff the args of IgnoreX claim to do so."""
    y, X = load_longley()

    fcst = MagicMock()

    if ignore_x is None:
        igx = IgnoreX(forecaster=fcst)
    else:
        igx = IgnoreX(forecaster=fcst, ignore_x=ignore_x)

    igx.fit(y, fh=[1, 2, 3], X=X)

    mock_fitted = igx.forecaster_.fit
    call_args_list = mock_fitted.call_args_list
    all_calls_X_none = all(call_args[1]["X"] is None for call_args in call_args_list)
    any_calls_X_none = any(call_args[1]["X"] is None for call_args in call_args_list)

    if ignore_x in [True, None]:
        assert all_calls_X_none
    else:
        assert not any_calls_X_none


@pytest.mark.skipif(
    not run_test_for_class(IgnoreX),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ignoreX_predict_with_longer_X():
    """Test predict accepts longer X when ignore_x=True by ignoring X."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=3)

    # Add one extra row so predict receives X longer than fh.
    X_extra = X_test.iloc[[-1]].copy()
    X_extra.index = pd.Index([X_test.index[-1] + 1])
    X_test_longer = pd.concat([X_test, X_extra])

    igx = IgnoreX(forecaster=NaiveForecaster(), ignore_x=True)
    igx.fit(y=y_train, X=X_train, fh=y_test.index)
    y_pred = igx.predict(X=X_test_longer)

    assert len(y_pred) == len(y_test)
