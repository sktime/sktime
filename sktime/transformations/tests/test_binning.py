"""Tests for TimeBinAggregate.

Pins the fix for sktime/sktime#10173:
``TimeBinAggregate(return_index='bin_mid')`` used to raise ``IndexError``
because the comprehension iterated ``range(len(bins))`` while indexing
``bins[i + 1]``.

The tests below exercise ``_transform`` directly to focus on the
off-by-one fix without involving sktime's input/output mtype checks
(which are a separate, pre-existing concern for this transformer's
``bin_mid`` mode — the resulting float index is not currently in
sktime's supported mtype set, but that is orthogonal to the
``IndexError`` regression).
"""

__author__ = ["jbbqqf"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.binning import TimeBinAggregate


def _toy_df():
    """8-row frame with a default RangeIndex 0..7."""
    return pd.DataFrame({"y": np.arange(8.0)})


@pytest.mark.skipif(
    not run_test_for_class(TimeBinAggregate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_bin_mid_does_not_raise_indexerror():
    """``return_index='bin_mid'`` must not crash on the last iteration.

    On ``origin/main`` this raises ``IndexError: list index out of range``
    because the comprehension reads ``bins[len(bins)]``.
    """
    transformer = TimeBinAggregate(bins=[0, 2, 4, 6, 8], return_index="bin_mid")
    transformer.fit(_toy_df())
    out = transformer._transform(_toy_df(), y=None)
    # Four bins → four mid-points: (0+2)/2, (2+4)/2, (4+6)/2, (6+8)/2.
    assert list(out.index) == [1.0, 3.0, 5.0, 7.0]


@pytest.mark.skipif(
    not run_test_for_class(TimeBinAggregate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "return_index,expected_index",
    [
        ("bin_start", [0, 2, 4, 6]),
        ("bin_end", [2, 4, 6, 8]),
        ("bin_mid", [1.0, 3.0, 5.0, 7.0]),
    ],
)
def test_return_index_modes_have_consistent_length(return_index, expected_index):
    """``bin_start``/``bin_end``/``bin_mid`` produce one row per bin (4)."""
    transformer = TimeBinAggregate(bins=[0, 2, 4, 6, 8], return_index=return_index)
    transformer.fit(_toy_df())
    out = transformer._transform(_toy_df(), y=None)
    assert len(out.index) == 4
    assert list(out.index) == expected_index


@pytest.mark.skipif(
    not run_test_for_class(TimeBinAggregate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_time_bin_aggregate_bin_mid_uses_midpoints():
    """Test that return_index='bin_mid' uses bin midpoints as output index."""
    X = pd.DataFrame({"y": [1, 2, 3, 4]}, index=[0, 1, 2, 3])

    transformer = TimeBinAggregate(bins=[0, 2, 4], return_index="bin_mid")
    Xt = transformer.fit_transform(X)

    assert list(Xt.index) == [1, 3]


def _value_range(values):
    """A scalar-valued aggregation for one variable, including empty bins."""
    return values.max() - values.min()


@pytest.mark.skipif(
    not run_test_for_class(TimeBinAggregate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("as_interval_index", [False, True])
@pytest.mark.parametrize(
    "return_index", ["range", "bin_start", "bin_end", "bin_mid", "bin"]
)
@pytest.mark.parametrize(
    "aggfunc,expected_values",
    [
        (None, [[2.0, 102.0], [np.nan, np.nan], [6.0, 106.0]]),
        (np.sum, [[4.0, 204.0], [0.0, 0.0], [12.0, 212.0]]),
        (_value_range, [[2.0, 2.0], [np.nan, np.nan], [2.0, 2.0]]),
    ],
)
def test_aggregate_each_variable(
    as_interval_index, return_index, aggfunc, expected_values
):
    """Aggregate columns independently and keep empty bins and variable names."""
    X = pd.DataFrame(
        {"low": [1.0, 3.0, 5.0, 7.0], "high": [101.0, 103.0, 105.0, 107.0]},
        index=[0, 1, 4, 5],
    )
    original = X.copy(deep=True)
    breaks = [-1, 1, 3, 5]
    intervals = pd.IntervalIndex.from_breaks(breaks)
    bins = intervals if as_interval_index else breaks
    expected_index = {
        "range": pd.RangeIndex(3),
        "bin_start": pd.Index([-1, 1, 3]),
        "bin_end": pd.Index([1, 3, 5]),
        "bin_mid": pd.Index([0, 2, 4]),
        "bin": intervals,
    }[return_index]
    expected = pd.DataFrame(expected_values, index=expected_index, columns=X.columns)

    transformer = TimeBinAggregate(
        bins=bins, aggfunc=aggfunc, return_index=return_index
    )
    if return_index == "bin":
        # IntervalIndex is not supported by BaseTransformer's output mtype check.
        # Exercise aggregation directly for this mode; the other modes use the API.
        transformer.fit(X)
        result = transformer._transform(X)
    else:
        result = transformer.fit_transform(X)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.skipif(
    not run_test_for_class(TimeBinAggregate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("aggfunc,values", [(None, [2.0, 6.0]), (np.sum, [4.0, 12.0])])
def test_aggregate_named_series(aggfunc, values):
    """A univariate input keeps its Series name after aggregation."""
    X = pd.Series([1.0, 3.0, 5.0, 7.0], name="signal")
    result = TimeBinAggregate(
        bins=[-1, 1, 3], aggfunc=aggfunc, return_index="range"
    ).fit_transform(X)
    expected = pd.Series(values, name="signal")
    pd.testing.assert_series_equal(result, expected)
