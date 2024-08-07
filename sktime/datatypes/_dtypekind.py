"""Class definition of DtypeKind and related utility functions.

Enum values of DtypeKind are in accordance with the dataframe interchange protocol.
Utility functions to help convert dtypes to corresponding DtypeKind values.
"""

__author__ = ["Abhay-Lejith", "pranavvp16"]

from enum import IntEnum

from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_signed_integer_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
    is_unsigned_integer_dtype,
)


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


def _pandas_dtype_to_kind(col_dtypes):
    """Convert pandas dtypes to enum DtypeKind values."""
    if isinstance(col_dtypes, list):
        return [_pandas_dtype_to_kind(x) for x in col_dtypes]

    if is_float_dtype(col_dtypes):
        return DtypeKind.FLOAT
    elif is_signed_integer_dtype(col_dtypes):
        return DtypeKind.INT
    elif is_unsigned_integer_dtype(col_dtypes):
        return DtypeKind.UINT
    elif col_dtypes == "object" or col_dtypes == "category":
        return DtypeKind.CATEGORICAL
    elif is_bool_dtype(col_dtypes):
        return DtypeKind.BOOL
    elif is_datetime64_any_dtype(col_dtypes) or is_timedelta64_dtype(col_dtypes):
        return DtypeKind.DATETIME
    elif is_string_dtype(col_dtypes):
        return DtypeKind.STRING
    else:
        raise TypeError(
            "Dtype of columns can be: INT, UINT, FLOAT, BOOL, STRING, "
            f"DATETIME, CATEGORICAL. Found dtype: {col_dtypes}"
        )


def _polars_dtype_to_kind(col_dtypes):
    """Convert polars dtypes to enum DtypeKind values."""
    import polars as pl

    if isinstance(col_dtypes, list):
        return [_polars_dtype_to_kind(x) for x in col_dtypes]

    if col_dtypes.is_float():
        return DtypeKind.FLOAT
    elif col_dtypes.is_signed_integer():
        return DtypeKind.INT
    elif col_dtypes.is_unsigned_integer():
        return DtypeKind.UINT
    elif isinstance(col_dtypes, pl.Categorical) or isinstance(col_dtypes, pl.Object):
        return DtypeKind.CATEGORICAL
    elif isinstance(col_dtypes, pl.Boolean):
        return DtypeKind.BOOL
    elif col_dtypes.is_temporal():
        return DtypeKind.DATETIME
    elif isinstance(col_dtypes, pl.String):
        return DtypeKind.STRING
    else:
        raise TypeError(
            "Dtype of columns can be: INT, UINT, FLOAT, BOOL, STRING, "
            f"DATETIME, CATEGORICAL. Found dtype: {col_dtypes}"
        )


# function for series scitype
def _get_series_dtypekind(obj, mtype):
    if mtype in ["numpy", "xarray"]:
        if len(obj.shape) == 2:
            return [DtypeKind.FLOAT] * obj.shape[1]
        else:
            return [DtypeKind.FLOAT]
    elif mtype == "pd.Series":
        col_dtypes = [obj.dtypes]
    elif mtype == "pd.DataFrame":
        col_dtypes = obj.dtypes.to_list()

    col_DtypeKinds = _pandas_dtype_to_kind(col_dtypes)

    return col_DtypeKinds


def _get_panel_dtypekind(obj, mtype):
    if mtype == "numpy3D":
        return [DtypeKind.FLOAT] * obj.shape[1]
    elif mtype == "numpyflat":
        return [DtypeKind.FLOAT]
    elif mtype == "df-list":
        return _get_series_dtypekind(obj[0], "pd.DataFrame")
    elif mtype == "pd-multiindex":
        col_dtypes = obj.dtypes.to_list()
    elif mtype == "nested_univ":
        col_names = obj.columns.to_list()
        col_dtypes = [obj[col].iloc[0].dtype for col in col_names[:-1]]
        # handling case where in dataloaders if X/y is not split, nested_univ may
        # contain str type in last column(y) which does not have dtype attr.
        # therefore, checking last column separately.
        last_col = col_names[-1]
        if isinstance(obj[last_col].iloc[0], str):
            col_dtypes.append(str)
        else:
            col_dtypes.append(obj[last_col].iloc[0].dtype)

    col_DtypeKinds = _pandas_dtype_to_kind(col_dtypes)

    return col_DtypeKinds


def _get_table_dtypekind(obj, mtype):
    if mtype == "numpy1D":
        return [DtypeKind.FLOAT]
    elif mtype == "numpy2D":
        return [DtypeKind.FLOAT] * obj.shape[1]
    elif mtype == "pd.Series":
        col_dtypes = [obj.dtypes]
    elif mtype == "pd.DataFrame":
        col_dtypes = obj.dtypes.to_list()
    elif mtype == "list_of_dict":
        col_dtypes = [type(obj[0][key]) for key in obj[0].keys()]

    col_Dtypekinds = _pandas_dtype_to_kind(col_dtypes)

    return col_Dtypekinds


# This function is to broadly classify all dtypekinds into CATEGORICAL or FLOAT
def _get_feature_kind(col_dtypekinds):
    feature_kind_map = {
        DtypeKind.CATEGORICAL: DtypeKind.CATEGORICAL,
        DtypeKind.STRING: DtypeKind.CATEGORICAL,
        DtypeKind.DATETIME: DtypeKind.CATEGORICAL,
        DtypeKind.BOOL: DtypeKind.FLOAT,
        DtypeKind.FLOAT: DtypeKind.FLOAT,
        DtypeKind.INT: DtypeKind.FLOAT,
        DtypeKind.UINT: DtypeKind.FLOAT,
    }

    feature_kind = []
    for kind in col_dtypekinds:
        feature_kind.append(feature_kind_map[kind])

    return feature_kind
