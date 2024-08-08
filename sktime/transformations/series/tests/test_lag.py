"""Tests for Lag transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["fkiraly"]

import itertools

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import get_examples
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.lag import Lag
from sktime.utils._testing.series import _make_series

# some examples with range vs time index, univariate vs multivariate (mv)
X_range_idx = get_examples("pd.DataFrame")[0]
X_range_idx_mv = get_examples("pd.DataFrame")[1]
X_time_idx = _make_series()
X_time_idx_mv = _make_series(n_columns=2)

# all fixtures
X_fixtures = [X_range_idx, X_range_idx_mv, X_time_idx, X_time_idx_mv]

# fixtures with time index
X_time_fixtures = [X_time_idx, X_time_idx_mv]

index_outs = ["original", "extend", "shift"]


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("X", X_fixtures)
@pytest.mark.parametrize("index_out", index_outs)
def test_lag_fit_transform_out_index(X, index_out):
    """Test that index sets of fit_transform output behave as expected."""
    t = Lag(2, index_out=index_out)
    Xt = t.fit_transform(X)

    if index_out == "original":
        assert Xt.index.equals(X.index)
    elif index_out == "extend":
        assert X.index.isin(Xt.index).all()
        assert len(Xt) == len(X) + 2
    elif index_out == "shift":
        assert len(Xt) == len(X)
        assert X.index[2:].isin(Xt.index).all()
        assert not X.index[:2].isin(Xt.index).any()


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("X", X_fixtures)
@pytest.mark.parametrize("index_out", index_outs)
@pytest.mark.parametrize("remember_data", [True, False])
def test_lag_fit_transform_out_values(X, index_out, remember_data):
    """Test that index sets of fit_transform output behave as expected."""
    t = Lag(2, index_out=index_out, remember_data=remember_data)
    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)
    X_fit = X[:2]
    X_trafo = X[2:]
    Xt = t.fit(X_fit).transform(X_trafo)

    if index_out in ["original", "extend"]:
        if remember_data:
            np.testing.assert_equal(Xt.iloc[0].values, X_fit.iloc[0].values)
        else:  # remember_data == False
            assert all(Xt.iloc[0].isna().values)
        if len(Xt) > 2:
            np.testing.assert_equal(Xt.iloc[2].values, X_trafo.iloc[0].values)

    elif index_out == "shift":
        np.testing.assert_equal(Xt.values, X_trafo.values)


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("X", X_fixtures)
@pytest.mark.parametrize("index_out", index_outs)
@pytest.mark.parametrize("lag", [2, [2, 4], [-1, 0, 5]])
def test_lag_fit_transform_columns(X, index_out, lag):
    """Test that columns of fit_transform output behave as expected."""
    t = Lag(lags=lag, index_out=index_out)
    Xt = t.fit_transform(X)

    if isinstance(lag, list):
        len_lag = len(lag)
    else:
        len_lag = 1

    def ncols(obj):
        if isinstance(obj, pd.DataFrame):
            return len(obj.columns)
        else:
            return 1

    assert ncols(Xt) == ncols(X) * len_lag


@pytest.mark.skipif(
    not run_test_for_class(Lag),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("X", X_fixtures)
@pytest.mark.parametrize("index_out", index_outs)
@pytest.mark.parametrize("lags", [2, [2, 4]])
def test_lag_fit_transform_column_names(X, index_out, lags):
    """Test expected column names."""
    t = Lag(lags=lags, index_out=index_out)
    Xt = t.fit_transform(X)

    if isinstance(Xt, pd.DataFrame):
        lag_col_names = set(Xt.columns)

        if isinstance(X, pd.DataFrame):
            col_names = X.columns
        elif isinstance(X, pd.Series):
            col_names = [X.name if X.name else 0]
        else:
            pass

        lags = [lags] if isinstance(lags, int) else lags
        expected = {
            f"lag_{lag}__{col}" for lag, col in itertools.product(lags, col_names)
        }
        assert lag_col_names == expected

    elif isinstance(Xt, pd.Series):
        assert Xt.name is None
