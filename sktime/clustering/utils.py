# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["convert_df_to_sklearn_format"]


from typing import Callable, List
import numpy as np

from sktime.clustering.types import (
    Data_Frame,
    Data_Parameter_Arr,
    Numpy_Array,
    Series,
    Data_Parameter,
    Tuple_Of_Numpy,
)


def convert_df_to_sklearn_format(df: Data_Frame) -> Numpy_Array:
    """
    Method that is used to convert the sktime dataframe into a format that
    can be passed into sklearn algorithms

    Parameters
    ----------
    df: sktime dataframe
        Sktime dataframe to be converted into sklearn format

    Returns
    -------
    sklearn_format: 2D numpy array
        Numpy array ready to be passed into sklearn algorithms
    """
    find_longest_series: Callable[[Series], int] = lambda series: len(series)

    for col in df:
        max_length_series: int = df[col].map(find_longest_series).max()
        __check_shape(df[col], max_length_series)

    sklearn_format = np.concatenate(df.fillna("").values.tolist())
    return sklearn_format


def __check_shape(series: Series, max_length: int) -> Numpy_Array:
    """
    Method that is used to check the shape of the data frame to ensure
    no uneven length or missing values. Desirable to flag this to the
    user as early as possible

    TODO: Add more rigerous testing of series length and throw more
    informative excpetions

    Parameters
    ----------
    series: Series
        Pandas series that contains sub series i.e.
        (series[series1, series2, ..., seriesn])
    max_length: int
        Integer that is the intended max length of each array. Needed
        so that the array can be padded to the correct legnth
    """
    for sub in series:
        sub_series_len: int = sub.shape[0]
        if sub_series_len != max_length:
            raise DataFormatError(
                "Cannot convert df as not all \
                            series are equal length"
            )


def __check_array_type(arr: Numpy_Array, arr_type: any):
    """
    Method that ensures the dtype of the numpy array is what is desired.
    If it is not an error is thrown.

    Parameters
    ----------
    arr: Numpy_Array
        Array to check the type of
    arr_type: any
        Numpy array type

    Retunrs
    -------
    boolean:
        true if same data type, false if different datatypes
    """
    if arr.dtype == arr_type:
        return True
    raise DataFormatError("Numpy array is not of type", arr_type)


class DataFormatError(Exception):
    """
    Exception class to informa the user about a formatting error related
    to the array passed

    Attributes
    ---------
    position: int
        index position that the error was detected
    message: str
        String explanation of the error
    """

    def __init__(self, message: str, position: int = -1):
        self.position = position
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.position != -1:
            return (
                "The array provided is formatted incorrectly due to "
                f"{self.message}. \n\n The error was detected at index "
                f"position: {self.position}"
            )
        return "The array provided is formatted incorrectly due to " f"{self.message}."


def check_data_parameters(data: Data_Parameter) -> Numpy_Array:
    """
    Method that is used to check the form of the data being passed and if
    it is in the sktime dataframe, it is converted to a numpy

    Parameters
    ----------
    data: Data_Parameter (numpy array or data frame)
        Data being passed to the cluster model
    """
    if isinstance(data, Numpy_Array):
        return data
    elif isinstance(data, Data_Frame):
        return convert_df_to_sklearn_format(data)
    else:
        raise ValueError(
            "The data passsed must be either a numpy array \
            or a data frame"
        )


def check_multiple_data_parameters(data_arr: Data_Parameter_Arr) -> Tuple_Of_Numpy:
    """
    Method used to check multiple data parameters

    Parameters
    ----------
    data_arr: Data_Parameter_Arr
        Array of Data_Parameters which are either numpy arrays or dataframes

    Returns
    -------
    x1, x2,..., xn: Numpy_Array
    """
    data_vals: List[Numpy_Array] = []
    for arr in data_arr:
        data_vals.append(check_data_parameters(arr))
    return tuple(data_vals)
