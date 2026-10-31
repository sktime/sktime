"""Machine type converters for Table scitype.

Exports conversion and mtype dictionary for Table scitype:

convert_dict: dict indexed by triples of str
  1st element = convert from - str
  2nd element = convert to - str
  3rd element = considered as this scitype - str
elements are conversion functions of machine type (1st) -> 2nd

Function signature of all elements
convert_dict[(from_type, to_type, as_scitype)]

Parameters
----------
obj : from_type - object to convert
store : dictionary - reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

Raises
------
ValueError and TypeError, if requested conversion is not possible
                            (depending on conversion logic)
"""

__author__ = ["fkiraly", "shlok191"]

__all__ = ["convert_dict"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datatypes._base import BaseConverter

# this needs to be refactored with the convert module
MTYPE_LIST_TABLE = [
    "pd_DataFrame_Table",
    "numpy1D",
    "numpy2D",
    "pd_Series_Table",
    "list_of_dict",
    "polars_eager_table",
    "polars_lazy_table",
]

##############################################################
# methods to convert one machine type to another machine type
##############################################################


##############################################################
# converter classes, one class per conversion
##############################################################


class TableIdentity(BaseConverter):
    """All Table scitype conversions of an mtype to itself."""

    _tags = {
        "object_type": "converter",
        "mtype_from": None,
        "mtype_to": None,
        "multiple_conversions": True,
        "python_version": None,
        "python_dependencies": None,
    }

    @classmethod
    def get_conversions(cls):
        """Get all conversions.

        Returns
        -------
        list of tuples (str, str)
            List of all conversions in this class.
        """
        excluded = []
        return [(tp, tp) for tp in MTYPE_LIST_TABLE if tp not in excluded]

    def _convert(self, obj, store=None):
        return convert_identity(obj, store=store)


class TableNumpy1DToNumpy2D(BaseConverter):
    """Conversion: numpy1D -> numpy2D, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "numpy1D",
        "mtype_to": "numpy2D",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_1D_to_2D_numpy_as_Table(obj, store=store)


class TableNumpy2DToNumpy1D(BaseConverter):
    """Conversion: numpy2D -> numpy1D, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "numpy2D",
        "mtype_to": "numpy1D",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_2D_to_1D_numpy_as_Table(obj, store=store)


class TablePdDataFrameToNumpy2D(BaseConverter):
    """Conversion: pd_DataFrame_Table -> numpy2D, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "numpy2D",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_df_to_2Dnp_as_Table(obj, store=store)


class TablePdDataFrameToNumpy1D(BaseConverter):
    """Conversion: pd_DataFrame_Table -> numpy1D, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "numpy1D",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_df_to_1Dnp_as_Table(obj, store=store)


class TableNumpy2DToPdDataFrame(BaseConverter):
    """Conversion: numpy2D -> pd_DataFrame_Table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "numpy2D",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_2Dnp_to_df_as_Table(obj, store=store)


class TableNumpy1DToPdDataFrame(BaseConverter):
    """Conversion: numpy1D -> pd_DataFrame_Table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "numpy1D",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_1Dnp_to_df_as_Table(obj, store=store)


class TablePdSeriesToPdDataFrame(BaseConverter):
    """Conversion: pd_Series_Table -> pd_DataFrame_Table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_Series_Table",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_s_to_df_as_table(obj, store=store)


class TablePdDataFrameToPdSeries(BaseConverter):
    """Conversion: pd_DataFrame_Table -> pd_Series_Table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "pd_Series_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_df_to_s_as_table(obj, store=store)


class TableListOfDictToPdDataFrame(BaseConverter):
    """Conversion: list_of_dict -> pd_DataFrame_Table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "list_of_dict",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_list_of_dict_to_df_as_table(obj, store=store)


class TablePdDataFrameToListOfDict(BaseConverter):
    """Conversion: pd_DataFrame_Table -> list_of_dict, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "list_of_dict",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj, store=None):
        return convert_df_to_list_of_dict_as_table(obj, store=store)


class TablePdDataFrameToPolarsEager(BaseConverter):
    """Conversion: pd_DataFrame_Table -> polars_eager_table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "polars_eager_table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
    }

    def _convert(self, obj, store=None):
        return convert_pandas_to_polars_eager(obj, store=store)


class TablePdDataFrameToPolarsLazy(BaseConverter):
    """Conversion: pd_DataFrame_Table -> polars_lazy_table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "polars_lazy_table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
    }

    def _convert(self, obj, store=None):
        return convert_pandas_to_polars_lazy(obj, store=store)


class TablePolarsEagerToPdDataFrame(BaseConverter):
    """Conversion: polars_eager_table -> pd_DataFrame_Table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "polars_eager_table",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
    }

    def _convert(self, obj, store=None):
        return convert_polars_to_pandas(obj, store=store)


class TablePolarsLazyToPdDataFrame(BaseConverter):
    """Conversion: polars_lazy_table -> pd_DataFrame_Table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "polars_lazy_table",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
    }

    def _convert(self, obj, store=None):
        return convert_polars_to_pandas(obj, store=store)


class TablePolarsLazyToPolarsEager(BaseConverter):
    """Conversion: polars_lazy_table -> polars_eager_table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "polars_lazy_table",
        "mtype_to": "polars_eager_table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
    }

    def _convert(self, obj, store=None):
        return convert_polars_lazy_to_eager(obj, store=store)


class TablePolarsEagerToPolarsLazy(BaseConverter):
    """Conversion: polars_eager_table -> polars_lazy_table, of Table scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "polars_eager_table",
        "mtype_to": "polars_lazy_table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
    }

    def _convert(self, obj, store=None):
        return convert_polars_eager_to_lazy(obj, store=store)


convert_dict = dict()


def convert_identity(obj, store=None):
    return obj


# assign identity converter to type conversion to self
for tp in MTYPE_LIST_TABLE:
    if (tp, tp) in TableIdentity.get_conversions():
        convert_dict[(tp, tp, "Table")] = TableIdentity(tp, tp)
    else:
        convert_dict[(tp, tp, "Table")] = convert_identity


def convert_1D_to_2D_numpy_as_Table(obj: np.ndarray, store=None) -> np.ndarray:
    if not isinstance(obj, np.ndarray):
        raise TypeError("input must be a np.ndarray")

    if len(obj.shape) == 1:
        res = np.reshape(obj, (-1, 1))
    else:
        raise TypeError("input must be 1D np.ndarray")

    return res


convert_dict[("numpy1D", "numpy2D", "Table")] = TableNumpy1DToNumpy2D()


def convert_2D_to_1D_numpy_as_Table(obj: np.ndarray, store=None) -> np.ndarray:
    if not isinstance(obj, np.ndarray):
        raise TypeError("input must be a np.ndarray")

    if len(obj.shape) == 2:
        res = obj.flatten()
    else:
        raise TypeError("input must be 2D np.ndarray")

    return res


convert_dict[("numpy2D", "numpy1D", "Table")] = TableNumpy2DToNumpy1D()


def convert_df_to_2Dnp_as_Table(obj: pd.DataFrame, store=None) -> np.ndarray:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input must be a pd.DataFrame")

    if isinstance(store, dict):
        store["columns"] = obj.columns

    return obj.to_numpy()


convert_dict[("pd_DataFrame_Table", "numpy2D", "Table")] = TablePdDataFrameToNumpy2D()


def convert_df_to_1Dnp_as_Table(obj: pd.DataFrame, store=None) -> np.ndarray:
    return convert_df_to_2Dnp_as_Table(obj=obj, store=store).flatten()


convert_dict[("pd_DataFrame_Table", "numpy1D", "Table")] = TablePdDataFrameToNumpy1D()


def convert_2Dnp_to_df_as_Table(obj: np.ndarray, store=None) -> pd.DataFrame:
    if not isinstance(obj, np.ndarray) and len(obj.shape) != 2:
        raise TypeError("input must be a 2D np.ndarray")

    if len(obj.shape) == 1:
        obj = np.reshape(obj, (-1, 1))

    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == obj.shape[1]
    ):
        res = pd.DataFrame(obj, columns=store["columns"])
    else:
        res = pd.DataFrame(obj)

    return res


convert_dict[("numpy2D", "pd_DataFrame_Table", "Table")] = TableNumpy2DToPdDataFrame()


def convert_1Dnp_to_df_as_Table(obj: np.ndarray, store=None) -> pd.DataFrame:
    if not isinstance(obj, np.ndarray) and len(obj.shape) != 1:
        raise TypeError("input must be a 1D np.ndarray")

    obj = np.reshape(obj, (-1, 1))

    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == obj.shape[1]
    ):
        res = pd.DataFrame(obj, columns=store["columns"])
    else:
        res = pd.DataFrame(obj)

    return res


convert_dict[("numpy1D", "pd_DataFrame_Table", "Table")] = TableNumpy1DToPdDataFrame()


def convert_s_to_df_as_table(obj: pd.Series, store=None) -> pd.DataFrame:
    if not isinstance(obj, pd.Series):
        raise TypeError("input must be a pd.Series")

    if (
        isinstance(store, dict)
        and "columns" in store.keys()
        and len(store["columns"]) == 1
    ):
        res = pd.DataFrame(obj, columns=store["columns"])
    else:
        res = pd.DataFrame(obj)

    return res


convert_dict[("pd_Series_Table", "pd_DataFrame_Table", "Table")] = (
    TablePdSeriesToPdDataFrame()
)


def convert_df_to_s_as_table(obj: pd.DataFrame, store=None) -> pd.Series:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input is not a pd.DataFrame")

    if len(obj.columns) != 1:
        raise ValueError("input must be univariate pd.DataFrame, with one column")

    if isinstance(store, dict):
        store["columns"] = obj.columns[[0]]

    y = obj[obj.columns[0]]
    y.name = None

    return y


convert_dict[("pd_DataFrame_Table", "pd_Series_Table", "Table")] = (
    TablePdDataFrameToPdSeries()
)


def convert_list_of_dict_to_df_as_table(obj: list, store=None) -> pd.DataFrame:
    if not isinstance(obj, list):
        raise TypeError("input must be a list of dict")

    if not np.all([isinstance(x, dict) for x in obj]):
        raise TypeError("input must be a list of dict")

    res = pd.DataFrame(obj)

    if (
        isinstance(store, dict)
        and "index" in store.keys()
        and len(store["index"]) == len(res)
    ):
        res.index = store["index"]

    return res


convert_dict[("list_of_dict", "pd_DataFrame_Table", "Table")] = (
    TableListOfDictToPdDataFrame()
)


def convert_df_to_list_of_dict_as_table(obj: pd.DataFrame, store=None) -> list:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input is not a pd.DataFrame")

    ret_dict = [obj.loc[i].to_dict() for i in obj.index]

    if isinstance(store, dict):
        store["index"] = obj.index

    return ret_dict


convert_dict[("pd_DataFrame_Table", "list_of_dict", "Table")] = (
    TablePdDataFrameToListOfDict()
)


if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    import polars as pl

    def convert_polars_to_pandas(obj, store=None):
        if not isinstance(obj, (pl.LazyFrame, pl.DataFrame)):
            raise TypeError("input is not a polars frame")

        if isinstance(obj, pl.LazyFrame):
            obj = obj.collect()

        return obj.to_pandas()

    def convert_pandas_to_polars_eager(obj: pd.DataFrame, store=None):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("input is not a pd.DataFrame")

        return pl.DataFrame(obj)

    def convert_pandas_to_polars_lazy(obj: pd.DataFrame, store=None):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("input is not a pd.DataFrame")

        return pl.LazyFrame(obj)

    def convert_polars_eager_to_lazy(obj: pl.DataFrame, store=None) -> pl.LazyFrame:
        if not isinstance(obj, pl.DataFrame):
            raise TypeError("input is not a pl.DataFrame")

        return obj.lazy()

    def convert_polars_lazy_to_eager(obj: pl.LazyFrame, store=None) -> pl.DataFrame:
        if not isinstance(obj, pl.LazyFrame):
            raise TypeError("input is not a pl.LazyFrame")

        return obj.collect()

    convert_dict[("pd_DataFrame_Table", "polars_eager_table", "Table")] = (
        TablePdDataFrameToPolarsEager()
    )
    convert_dict[("pd_DataFrame_Table", "polars_lazy_table", "Table")] = (
        TablePdDataFrameToPolarsLazy()
    )

    convert_dict[("polars_eager_table", "pd_DataFrame_Table", "Table")] = (
        TablePolarsEagerToPdDataFrame()
    )
    convert_dict[("polars_lazy_table", "pd_DataFrame_Table", "Table")] = (
        TablePolarsLazyToPdDataFrame()
    )

    convert_dict[("polars_lazy_table", "polars_eager_table", "Table")] = (
        TablePolarsLazyToPolarsEager()
    )
    convert_dict[("polars_eager_table", "polars_lazy_table", "Table")] = (
        TablePolarsEagerToPolarsLazy()
    )
