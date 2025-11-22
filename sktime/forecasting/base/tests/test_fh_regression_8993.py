# regression tests for issue #8993

from sktime.forecasting.base import ForecastingHorizon
import pandas as pd


def test_rangeindex_from_series_is_absolute():
    """When FH initialized from a Series.index (RangeIndex) it must be absolute.

    Regression test for issue #8993.
    """
    ser = pd.Series(range(10))
    assert isinstance(ser.index, pd.RangeIndex)
    fh = ForecastingHorizon(ser.index)
    assert fh.is_relative is False


def test_list_of_ints_is_relative():
    """List of ints should be interpreted as relative FHs by default.

    This ensures we still treat lists e.g. [1,2,3] as relative, not absolute.
    """
    fh = ForecastingHorizon([1, 2, 3])
    assert fh.is_relative is True
