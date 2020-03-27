import pytest
import numpy as np
import pandas._testing as tm

from pandas import DataFrame, Series
from sktime.container import TimeFrame, TimeArray

from functools import partial
numpy64 = partial(np.array, dtype=np.int64)


class TestTimeFrameConstructors:
    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: TimeFrame(),
            lambda: TimeFrame(None),
            lambda: TimeFrame({}),
            lambda: TimeFrame(()),
            lambda: TimeFrame([]),
            lambda: TimeFrame((_ for _ in [])),
            lambda: TimeFrame(range(0)),
            lambda: TimeFrame(data=None),
            lambda: TimeFrame(data={}),
            lambda: TimeFrame(data=()),
            lambda: TimeFrame(data=[]),
            lambda: TimeFrame(data=(_ for _ in [])),
            lambda: TimeFrame(data=range(0)),
        ],
    )
    def test_empty_constructor(self, constructor):
        expected = TimeFrame()
        result = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data,columns",
        [
            ({'x': Series([1, 2, 3])}, None),
            ({'x': Series([1, 2, 3])}, ['x']),
            ({'x': numpy64([1, 2, 3])}, None),
            ({'x': numpy64([1, 2, 3])}, ['x']),
        ],
    )
    def test_base_pandas_1d(self, data, columns):
        df_equiv = DataFrame({'x': [1, 2, 3]})
        expected = TimeFrame(df_equiv)
        result = TimeFrame(data=data, columns=columns)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data,columns",
        [
            ({'x': Series([1, 4]), 'y': Series([2, 5]), 'z': Series([3, 6])}, None),
            ({'x': Series([1, 4]), 'y': Series([2, 5]), 'z': Series([3, 6])}, ['x', 'y', 'z']),
            ({'x': numpy64([1, 4]), 'y': numpy64([2, 5]), 'z': numpy64([3, 6])}, None),
            ({'x': numpy64([1, 4]), 'y': numpy64([2, 5]), 'z': numpy64([3, 6])}, ['x', 'y', 'z']),
            ([[1, 2, 3], [4, 5, 6]], ['x', 'y', 'z']),
            (numpy64([[1, 2, 3], [4, 5, 6]]), ['x', 'y', 'z'])
        ],
    )
    def test_no_timeseries_2d(self, data, columns):
        df_equiv = DataFrame({'x': [1, 4], 'y': [2, 5], 'z': [3, 6]})
        expected = TimeFrame(df_equiv)
        result = TimeFrame(data=data, columns=columns)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data,columns",
        [
            ({'x': numpy64([[1, 2, 3], [4, 5, 6]])}, None),
            ({'x': numpy64([[1, 2, 3], [4, 5, 6]])}, ['x']),
            (Series([Series([1, 2, 3]), Series([4, 5, 6])], name='x'), None),
            ({'x': Series([Series([1, 2, 3]), Series([4, 5, 6])])}, None),
            ({'x': DataFrame([[1, 2, 3], [4, 5, 6]])}, None),
            (DataFrame({'x': Series([Series([1, 2, 3]), Series([4, 5, 6])])}), None)
        ],
    )
    def test_sequence(self, data, columns):
        expected = TimeFrame(TimeArray(numpy64([[1, 2, 3], [4, 5, 6]])), columns=['x'])
        result = TimeFrame(data=data, columns=columns)
        assert np.all(expected['x'].time_index == result['x'].time_index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data,columns",
        [
            (numpy64([[[1, 2, 3], [4, 5, 6]], [[7, 4, 2], [5, 2, 1]]]), ['x', 'y']),
            (DataFrame({'x': Series([Series([1, 2, 3]), Series([4, 5, 6])]),
                        'y': Series([Series([7, 4, 2]), Series([5, 2, 1])])}), None)
        ],
    )
    def test_multi_column(self, data, columns):
        expected = TimeFrame({'x': TimeArray(numpy64([[1, 2, 3], [4, 5, 6]])),
                              'y': TimeArray(numpy64([[7, 4, 2], [5, 2, 1]]))})
        result = TimeFrame(data=data, columns=columns)
        assert np.all(expected['x'].time_index == result['x'].time_index)
        assert np.all(expected['y'].time_index == result['y'].time_index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "data,column,index",
        [
            (TimeArray(numpy64([[1, 2, 3], [4, 5, 6]])), 'col_1', None),
        ],
    )
    def test_type_error(self, data, column, index):
        with pytest.raises(TypeError) as e_info:
            TimeFrame(data, index, column)

    @pytest.mark.parametrize(
        "data,column,index",
        [
            (TimeArray(numpy64([[1, 2, 3], [4, 5, 6]])), ['col_1', 'col_2'], None),
            ({'x': 1, 'y': 1}, None, None),
            (numpy64(1), None, None)
        ],
    )
    def test_value_error(self, data, column, index):
        with pytest.raises(ValueError) as e_info:
            TimeFrame(data, index, column)
