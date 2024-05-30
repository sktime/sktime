from enum import IntEnum

import numpy as np
import pandas as pd


class DtypeKind(IntEnum):
    """
    Integer enum for data types.

    Attributes
    ----------
    INT : int
        Matches to signed integer data type.
    UINT : int
        Matches to unsigned integer data type.
    FLOAT : int
        Matches to floating point data type.
    BOOL : int
        Matches to boolean data type.
    STRING : int
        Matches to string data type (UTF-8 encoded).
    DATETIME : int
        Matches to datetime data type.
    CATEGORICAL : int
        Matches to categorical data type.
    """

    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


def _simplify_dtype(col_dtypes):
    for i, dtype in enumerate(col_dtypes):
        if dtype == float:
            col_dtypes[i] = DtypeKind.FLOAT
        elif dtype == int:
            col_dtypes[i] = DtypeKind.INT
        elif dtype == np.uint:
            col_dtypes[i] = DtypeKind.UINT
        elif dtype == object:
            col_dtypes[i] = DtypeKind.CATEGORICAL
        elif dtype == bool:
            col_dtypes[i] = DtypeKind.BOOL
        elif dtype == pd.DatetimeIndex:
            col_dtypes[i] = DtypeKind.DATETIME
    return col_dtypes


# function for series scitype
def _get_series_dtypekind(obj, mtype):
    if mtype == np.ndarray:
        if len(obj.shape) == 2:
            col_dtypes = [float] * obj.shape[1]
        else:
            col_dtypes = [float]
    elif mtype == pd.Series:
        col_dtypes = [obj.dtypes]
    elif mtype == pd.DataFrame:
        col_dtypes = obj.dtypes.to_list()

    col_DtypeKinds = _simplify_dtype(col_dtypes)

    return col_DtypeKinds
