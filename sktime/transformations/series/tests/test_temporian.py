"""Tests for TemporianTransformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["ianspektor", "javiber"]

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline, load_lynx
from sktime.datatypes import get_examples
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.temporian import TemporianTransformer


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_flat_univariate():
    """Tests basic function works on flat (non-indexed) univariate time series."""
    X = load_airline()[0:32]

    def function(evset):
        return evset["Number of airline passengers"] + 1

    transformer = TemporianTransformer(function=function)
    X_transformed = transformer.fit_transform(X=X)

    pd.testing.assert_series_equal(X_transformed, X + 1)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_panel():
    """Tests basic function works on Panels."""

    def function(evset):
        return evset + 1

    examples = get_examples("pd-multiindex", as_scitype="Panel").values()
    assert len(examples) > 0

    for X in examples:
        transformer = TemporianTransformer(function=function)
        X_transformed = transformer.fit_transform(X=X)
        pd.testing.assert_frame_equal(X_transformed, X + 1)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_hier():
    """Tests basic function works on Hierarchical."""

    def function(evset):
        return evset + 1

    examples = get_examples("pd_multiindex_hier", as_scitype="Hierarchical").values()
    assert len(examples) > 0

    for X in examples:
        transformer = TemporianTransformer(function=function)
        X_transformed = transformer.fit_transform(X=X)
        pd.testing.assert_frame_equal(X_transformed, X + 1)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_dtypes():
    """Tests basic function works on all dtypes."""

    def function(evset):
        return evset + 1

    X = pd.DataFrame(
        {
            str(dtype): np.zeros(10).astype(dtype)
            for dtype in [
                np.int32,
                np.int64,
                np.float32,
                np.float64,
                np.bool_,
            ]
        }
    )
    transformer = TemporianTransformer(function=function)
    X_transformed = transformer.fit_transform(X=X)
    pd.testing.assert_frame_equal(X_transformed, X + 1)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_compiled():
    """Tests function already compiled."""
    import temporian as tp

    def function(evset):
        return evset + 1

    X = get_examples("pd.Series", as_scitype="Series")[0]

    transformer = TemporianTransformer(function=function, compile=True)
    # assert the function was compiled
    assert transformer.function.is_tp_compiled

    X_transformed = transformer.fit_transform(X=X)
    pd.testing.assert_series_equal(X_transformed, X + 1)

    # test that an already compiled function down't cause errors
    transformer = TemporianTransformer(function=tp.compile(function), compile=True)
    assert transformer.function.is_tp_compiled
    X_transformed = transformer.fit_transform(X=X)
    pd.testing.assert_series_equal(X_transformed, X + 1)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multiple_output():
    """Tests errors when functions return incorrect types."""

    def function(evset):
        split = datetime(1950, 7, 16)
        return evset.before(split), evset.after(split)

    X = load_airline()

    transformer = TemporianTransformer(function=function)
    with pytest.raises(
        TypeError, match="Expected return type to be an EventSet but received"
    ):
        _ = transformer.fit_transform(X=X)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multiple_input():
    """Tests errors on incorrectly defined functions."""

    def function(evset, incorrect_param):
        return evset + 1

    X = load_airline()

    transformer = TemporianTransformer(function=function)
    with pytest.raises(TypeError, match="missing 1 required positional"):
        _ = transformer.fit_transform(X=X)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_change_sampling():
    """Tests error when changing the sampling."""
    import temporian as tp

    # in this dataset each row is a period of 1 year,
    # lagging for one year would remove the first row thus changing the sampling
    X = load_lynx()

    def function(evset):
        return evset.lag(tp.duration.days(365))

    transformer = TemporianTransformer(function=function, compile=True)
    with pytest.raises(ValueError):
        _ = transformer.fit_transform(X=X)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_complex_function():
    """Tests function already compiled."""
    import temporian as tp

    def function(evset):
        return evset.simple_moving_average(tp.duration.days(3 * 365)).resample(evset)

    X = load_lynx()

    transformer = TemporianTransformer(function=function, compile=True)
    result = transformer.fit_transform(X=X)
    assert result.iloc[0] == X.iloc[0]
    assert result.iloc[-1] == X.iloc[-3:].mean()
