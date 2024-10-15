"""Tests for TemporianTransformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).

__author__ = ["ianspektor", "javiber"]

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_solar
from sktime.datatypes import get_examples
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.series.temporian import TemporianTransformer


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_flat_univariate():
    """Tests basic function works on flat (non-indexed) univariate time series."""
    X = load_solar()[0:32]

    def function(evset):
        return evset["solar_gen"] + 1

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
    """Tests compiling the function."""
    import temporian as tp

    def function(evset):
        return evset + 1

    X = get_examples("pd.Series", as_scitype="Series")[0]

    true_compile = tp.compile

    with patch.object(tp, "compile") as compile_patch:
        compile_patch.side_effect = true_compile
        transformer = TemporianTransformer(function=function, compile=True)
        X_transformed = transformer.fit_transform(X=X)
        pd.testing.assert_series_equal(X_transformed, X + 1)
        compile_patch.assert_called_once_with(function)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_compile_already_compiled():
    """Tests compiling doesn't fail if the function was compiled already."""
    import temporian as tp

    @tp.compile
    def function(evset):
        return evset + 1

    X = get_examples("pd.Series", as_scitype="Series")[0]

    true_compile = tp.compile

    with patch.object(tp, "compile") as compile_patch:
        compile_patch.side_effect = true_compile
        transformer = TemporianTransformer(function, compile=True)
        X_transformed = transformer.fit_transform(X=X)
        pd.testing.assert_series_equal(X_transformed, X + 1)
        compile_patch.assert_called_once_with(function)


@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multiple_output():
    """Tests errors when functions return incorrect types."""

    def function(evset):
        split = datetime(1950, 7, 16)
        return evset.before(split), evset.after(split)

    X = load_solar()

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

    X = load_solar()

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

    # in this dataset each row is a period of 1/2 hour
    # lagging for one hour would remove the first two rows thus changing the sampling
    X = load_solar()

    def function(evset):
        return evset.lag(tp.duration.hours(1))

    transformer = TemporianTransformer(function=function, compile=True)
    with pytest.raises(
        ValueError,
        match="The resulting EventSet must have the same sampling as the input",
    ):
        _ = transformer.fit_transform(X=X)


@pytest.mark.xfail(reason="Known bug #7080.")
@pytest.mark.skipif(
    not run_test_for_class(TemporianTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_complex_function():
    import temporian as tp

    # in this dataset each row is a period of 1/2 hour

    def function(evset):
        return evset.simple_moving_average(tp.duration.days(1)).resample(evset)

    X = load_solar()

    transformer = TemporianTransformer(function=function, compile=True)
    result = transformer.fit_transform(X=X)
    assert result.iloc[0] == X.iloc[0]
    assert np.allclose(result.iloc[-1], X.iloc[-48:].mean())
