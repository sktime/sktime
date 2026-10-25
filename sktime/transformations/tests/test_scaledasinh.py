"""Tests for ScaledAsinhTransformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Karman580"]
__all__ = []

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.scaledasinh import ScaledAsinhTransformer
from sktime.utils._testing.panel import _make_panel

# columns with very different locations and scales
X_MULTI = pd.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 3.5, 2.5],
        "b": [1001.0, 1004.0, 1002.0, 1010.0, 1003.0, 1001.5, 1007.0],
        "c": [-50.0, 20.0, 0.0, -10.0, 35.0, 5.0, -2.0],
    }
)


@pytest.mark.skipif(
    not run_test_for_class(ScaledAsinhTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_scaledasinh_columns_transformed_independently():
    """Each column is transformed as if it were fitted on its own, see #11564."""
    t = ScaledAsinhTransformer().fit(X_MULTI)
    Xt = t.transform(X_MULTI)

    assert t.shift_parameter_asinh_.shape == (X_MULTI.shape[1],)
    assert t.scale_parameter_asinh_.shape == (X_MULTI.shape[1],)

    for col in X_MULTI.columns:
        Xt_alone = ScaledAsinhTransformer().fit_transform(X_MULTI[[col]])
        np.testing.assert_allclose(Xt[col].values, Xt_alone[col].values)


@pytest.mark.skipif(
    not run_test_for_class(ScaledAsinhTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_scaledasinh_other_columns_do_not_affect_column():
    """Changing one column does not change the output of another column."""
    X_changed = X_MULTI.assign(b=X_MULTI["b"] * 10 + 500, c=-X_MULTI["c"])

    Xt = ScaledAsinhTransformer().fit_transform(X_MULTI)
    Xt_changed = ScaledAsinhTransformer().fit_transform(X_changed)

    np.testing.assert_allclose(Xt["a"].values, Xt_changed["a"].values)


@pytest.mark.skipif(
    not run_test_for_class(ScaledAsinhTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_scaledasinh_univariate_against_formula():
    """Univariate output follows asinh((x - a) / b) from the docstring."""
    from scipy.stats import median_abs_deviation

    y = load_airline()
    factor = 1.6
    t = ScaledAsinhTransformer(mad_normalization_factor=factor)
    yt = t.fit_transform(y)

    a = np.median(y.values)
    b = median_abs_deviation(y.values) * factor
    expected = np.arcsinh((y.values - a) / b)

    assert isinstance(yt, pd.Series)
    np.testing.assert_allclose(yt.values, expected)
    np.testing.assert_allclose(t.shift_parameter_asinh_, [a])
    np.testing.assert_allclose(t.scale_parameter_asinh_, [b])


@pytest.mark.skipif(
    not run_test_for_class(ScaledAsinhTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "X",
    [
        load_airline(),
        X_MULTI,
        _make_panel(
            n_instances=3, n_columns=2, n_timepoints=10, return_mtype="pd-multiindex"
        ),
    ],
    ids=["series", "dataframe", "panel"],
)
def test_scaledasinh_inverse_transform_roundtrip(X):
    """inverse_transform recovers the input for Series, DataFrame and Panel."""
    t = ScaledAsinhTransformer().fit(X)
    Xt = t.transform(X)
    X_inv = t.inverse_transform(Xt)

    # sinh(asinh(z)) is exact up to floating point error; values here are <= ~1e3
    np.testing.assert_allclose(np.asarray(X_inv), np.asarray(X), rtol=1e-10)


@pytest.mark.skipif(
    not run_test_for_class(ScaledAsinhTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_scaledasinh_panel_instances_and_columns_independent():
    """For Panel input, each instance and column is transformed independently."""
    X = _make_panel(
        n_instances=3, n_columns=2, n_timepoints=10, return_mtype="pd-multiindex"
    )
    # shift the second variable far away, so pooling would show up in the output
    X.iloc[:, 1] = X.iloc[:, 1] * 100 + 1e4

    Xt = ScaledAsinhTransformer().fit_transform(X)

    for inst in X.index.get_level_values(0).unique():
        for col in X.columns:
            X_single = X.loc[inst, [col]]
            Xt_single = ScaledAsinhTransformer().fit_transform(X_single)
            np.testing.assert_allclose(Xt.loc[inst, col].values, Xt_single[col].values)
