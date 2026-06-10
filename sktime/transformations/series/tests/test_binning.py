"""Tests for time binning transformers."""

import pandas as pd

from sktime.transformations.series.binning import TimeBinAggregate


def test_time_bin_aggregate_bin_mid_uses_midpoints():
    """Test that return_index='bin_mid' uses bin midpoints as output index."""
    X = pd.DataFrame({"y": [1, 2, 3, 4]}, index=[0, 1, 2, 3])

    transformer = TimeBinAggregate(bins=[0, 2, 4], return_index="bin_mid")
    Xt = transformer.fit_transform(X)

    assert list(Xt.index) == [1, 3]
