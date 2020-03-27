import pytest
import numpy as np
import pandas as pd
import pandas._testing as tm

from sktime.container import TimeArray, TimeSeries

from functools import partial
numpy64 = partial(np.array, dtype=np.int64)


class TestTimeSeriesConstructors:
    @pytest.mark.parametrize(
        "constructor,check_index_type",
        [
            # NOTE: some overlap with test_constructor_empty but that test does not
            # test for None or an empty generator.
            # test_constructor_pass_none tests None but only with the index also
            # passed.
            (lambda: TimeSeries(), True),
            (lambda: TimeSeries(None), True),
            (lambda: TimeSeries({}), True),
            (lambda: TimeSeries(()), False),  # creates a RangeIndex
            (lambda: TimeSeries([]), False),  # creates a RangeIndex
            (lambda: TimeSeries((_ for _ in [])), False),  # creates a RangeIndex
            (lambda: TimeSeries(data=None), True),
            (lambda: TimeSeries(data={}), True),
            (lambda: TimeSeries(data=()), False),  # creates a RangeIndex
            (lambda: TimeSeries(data=[]), False),  # creates a RangeIndex
            (lambda: TimeSeries(data=(_ for _ in [])), False),  # creates a RangeIndex
        ],
    )
    def test_empty_constructor(self, constructor, check_index_type):
        with tm.assert_produces_warning(DeprecationWarning, check_stacklevel=False):
            expected = TimeSeries()
            result = constructor()

        assert len(result.index) == 0
        tm.assert_series_equal(result, expected, check_index_type=check_index_type)

    @pytest.mark.parametrize(
        "data,index",
        [
            ([1, 2, 3], None),
            (['a', 'b', 'c'], [0, 1, 2]),
            (np.array(['a', 'b', 'c']), None),
            (numpy64([1, 2, 3]), ['a', 'b', 'c']),
            (pd.Series([1, 2, 3]), None),
            (pd.Series([1, 2, 3]), ['a']),
        ],
    )
    def test_base_dtypes(self, data, index):
        expected = pd.Series(data, index)
        result = TimeSeries(data, index)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            numpy64([[1, 1, 1], [1, 2, 3]]),
            pd.Series([pd.Series([1, 1, 1]), pd.Series([1, 2, 3])]),
            pd.DataFrame([[1, 1, 1], [1, 2, 3]])
        ],
    )
    def test_time_dtypes(self, data):
        expected = TimeSeries(TimeArray(numpy64([[1, 1, 1], [1, 2, 3]])))
        result = TimeSeries(data)
        tm.assert_series_equal(result, expected)