#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests of Imputer functionality."""

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import get_examples
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.compose import TransformByLevel
from sktime.transformations.impute import Imputer
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.hierarchical import _make_hierarchical

y, X = make_forecasting_problem(make_X=True)

X.iloc[3, 0] = np.nan
X.iloc[3, 1] = np.nan
X.iloc[0, 1] = np.nan
X.iloc[-1, 1] = np.nan

y.iloc[3] = np.nan
y.iloc[0] = np.nan
y.iloc[-1] = np.nan

z = _make_hierarchical(hierarchy_levels=(3,), n_columns=3)

z.iloc[3] = np.nan
z.iloc[0] = np.nan
z.iloc[-1] = np.nan


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("forecaster", [None, "NaiveForecaster"])
@pytest.mark.parametrize("value", [None, 1])
@pytest.mark.parametrize("Z", [y, X, z])
@pytest.mark.parametrize(
    "method",
    [
        "drift",
        "linear",
        "nearest",
        "constant",
        "mean",
        "median",
        "backfill",
        "pad",
        "random",
        "forecaster",
    ],
)
def test_imputer(method, Z, value, forecaster):
    """Test univariate and multivariate Imputer with all methods."""
    if forecaster == "NaiveForecaster" or method == "forecaster":
        from sktime.forecasting.naive import NaiveForecaster

        forecaster = NaiveForecaster()

    value = 1 if method == "constant" else value
    t = Imputer(method=method, forecaster=forecaster, value=value)
    y_hat = t.fit_transform(Z)
    assert not y_hat.isnull().to_numpy().any()


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "method",
    [
        "linear",
        "nearest",
        "mean",
        "median",
        "backfill",
        "pad",
    ],
)
def test_impute_multiindex(method):
    """Test for data leakage in case of pd-multiindex data.

    Failure case in bug #6224
    """
    df = get_examples(mtype="pd-multiindex")[0].copy()
    df.iloc[:3, :] = np.nan  # instance 0 entirely missing
    df.iloc[3:4, :] = np.nan  # instance 1 first timepoint missing
    df.iloc[8:, :] = np.nan  # instance 2 last timepoint missing

    imp = Imputer(method=method)
    df_imp = imp.fit_transform(df)

    # instance 0 entirely missing, so it should remain missing
    assert np.array_equal(df.iloc[:3, :], df_imp.iloc[:3, :], equal_nan=True)

    # instance 1 and 2 should not have any missing values
    assert not df_imp.iloc[3:, :].isna().any().any()

    # test consistency between applying the imputer to every instance separately,
    # vs applying them to the panel
    imp_tbl = TransformByLevel(Imputer(method=method))
    df_imp_tbl = imp_tbl.fit_transform(df)
    assert np.array_equal(df_imp, df_imp_tbl, equal_nan=True)


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_imputer_forecaster_y():
    """Test that forecaster imputer works with y.

    Failure case in bug #5284.
    """
    from sklearn.linear_model import LinearRegression

    from sktime.datasets import load_airline
    from sktime.forecasting.compose import YfromX

    X = load_airline()
    y = load_airline()

    model_reg = YfromX(LinearRegression())
    model_reg.fit(X, y)
    transformer = Imputer(method="forecaster", forecaster=model_reg)

    transformer.fit(X=X, y=y)
    transformer.transform(X=X, y=y)


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["mean", "median", "nearest", "linear"])
def test_imputer_with_non_numeric_columns(method):
    """Imputer methods for numeric data pass through non-numeric columns."""
    X = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, np.nan], "b": ["x", "y", None, "z"]},
        index=pd.date_range("2000-01-01", periods=4, freq="D"),
    )
    Xt = Imputer(method=method).fit_transform(X)

    assert not Xt.isna().any().any()
    # numeric column is imputed as for a purely numeric frame
    expected = Imputer(method=method).fit_transform(X[["a"]])
    pd.testing.assert_frame_equal(Xt[["a"]], expected)
    # non-numeric column is filled by forward/backward fill
    assert Xt["b"].tolist() == ["x", "y", "y", "z"]


@pytest.mark.skipif(
    not run_test_for_class(Imputer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("method", ["mean", "median"])
def test_imputer_multiindex_with_non_numeric_columns(method):
    """Mean and median imputation by instance pass through non-numeric columns."""
    X = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0, 7.0],
            "b": ["x", None, "y", "z", "z", None],
        },
        index=pd.MultiIndex.from_product([[0, 1], [0, 1, 2]]),
    )
    Xt = Imputer(method=method).fit_transform(X)

    assert not Xt.isna().any().any()
    assert Xt["a"].tolist() == [1.0, 2.0, 3.0, 6.0, 5.0, 7.0]
    assert Xt["b"].tolist() == ["x", "x", "y", "z", "z", "z"]
