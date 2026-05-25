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

from sktime.transformations.series.binning import TimeBinAggregate


def _toy_df():
    """8-row frame with a default RangeIndex 0..7."""
    return pd.DataFrame({"y": np.arange(8.0)})


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
