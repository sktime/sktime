# -*- coding: utf-8 -*-
import pytest
import pandas as pd
import numpy as np
from sktime.clustering.utils import (
    convert_df_to_sklearn_format,
    Data_Frame,
    SkLearn_Data,
    Numpy_Array,
    __check_shape,
    __check_array_type,
    DataFormatError,
    Series,
)


def test_convert_df_to_learn_format(df: Data_Frame):
    sklearn_data: SkLearn_Data = convert_df_to_sklearn_format(df)
    assert isinstance(sklearn_data, Numpy_Array)
    for arr in sklearn_data:
        assert isinstance(arr, Numpy_Array)


def test_check_shape():
    sub_series_len: int = 10
    series: Series = __create_random_series(sub_series_len)
    try:
        __check_shape(series, sub_series_len)
    except Exception:
        pytest.fail("The shape is not expected (this test should pass)")

    with pytest.raises(DataFormatError):
        series: Series = __create_random_series(sub_series_len)
        __check_shape(series, sub_series_len + 1)


def test_check_array_type():
    arr: Numpy_Array = __create_random_numpy(10, "float64")
    try:
        __check_array_type(arr, "float64")
    except Exception:
        pytest.fail("The array type is not as expected " "(this test should pass)")

    with pytest.raises(DataFormatError):
        arr: Numpy_Array = __create_random_numpy(10, "float64")
        __check_array_type(arr, "int32")


def __create_random_numpy(length: int, arr_type: str) -> Numpy_Array:
    arr: Numpy_Array = np.random.randn(length)
    arr.astype(arr_type)
    return arr


def __create_random_series(sub_series_len: int) -> Series:
    return pd.Series(
        [
            pd.Series(np.random.randn(sub_series_len)),
            pd.Series(np.random.randn(sub_series_len)),
            pd.Series(np.random.randn(sub_series_len)),
            pd.Series(np.random.randn(sub_series_len)),
            pd.Series(np.random.randn(sub_series_len)),
        ]
    )
