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
from sklearn.cluster import KMeans


def test_create_sklearn_k_means(df_x: Data_Frame, df_y: Data_Frame):
    sklearn_train_data: SkLearn_Data = convert_df_to_sklearn_format(df_x)
    km = KMeans(
        n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0
    )
    km.fit(sklearn_train_data)
    # print(y_km)


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
