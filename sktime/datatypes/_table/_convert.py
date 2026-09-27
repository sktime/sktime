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
from sktime.datatypes._convert_utils._convert import _extend_conversions

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

convert_dict = dict()


class TableIdentity(BaseConverter):
    """Identity converter for Table mtypes."""

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
        return [(tp, tp) for tp in MTYPE_LIST_TABLE]

    def _convert(self, obj, store=None):
        return obj


class Numpy1DToNumpy2D(BaseConverter):
    """Convert numpy1D to numpy2D."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "numpy1D",
        "mtype_to": "numpy2D",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: np.ndarray, store=None) -> np.ndarray:
        if not isinstance(obj, np.ndarray):
            raise TypeError("input must be a np.ndarray")

        if len(obj.shape) == 1:
            res = np.reshape(obj, (-1, 1))
        else:
            raise TypeError("input must be 1D np.ndarray")

        return res


class Numpy2DToNumpy1D(BaseConverter):
    """Convert numpy2D to numpy1D."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "numpy2D",
        "mtype_to": "numpy1D",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: np.ndarray, store=None) -> np.ndarray:
        if not isinstance(obj, np.ndarray):
            raise TypeError("input must be a np.ndarray")

        if len(obj.shape) == 2:
            res = obj.flatten()
        else:
            raise TypeError("input must be 2D np.ndarray")

        return res


class PandasDataFrameToNumpy2D(BaseConverter):
    """Convert pd_DataFrame_Table to numpy2D."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "numpy2D",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.DataFrame, store=None) -> np.ndarray:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("input must be a pd.DataFrame")

        if isinstance(store, dict):
            store["columns"] = obj.columns

        return obj.to_numpy()


class PandasDataFrameToNumpy1D(BaseConverter):
    """Convert pd_DataFrame_Table to numpy1D."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "numpy1D",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.DataFrame, store=None) -> np.ndarray:
        converter = PandasDataFrameToNumpy2D(
            mtype_from="pd_DataFrame_Table",
            mtype_to="numpy2D",
        )
        return converter(obj, store=store).flatten()


class Numpy2DToPandasDataFrame(BaseConverter):
    """Convert numpy2D to pd_DataFrame_Table."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "numpy2D",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: np.ndarray, store=None) -> pd.DataFrame:
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


class Numpy1DToPandasDataFrame(BaseConverter):
    """convert numpy1D to pd_DataFrame_Table."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "numpy1D",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: np.ndarray, store=None) -> pd.DataFrame:
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


class PandasSeriesToPandasDataFrame(BaseConverter):
    """convert pd_Series_Table to pd_DataFrame_Table."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_Series_Table",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.Series, store=None) -> pd.DataFrame:
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
    PandasSeriesToPandasDataFrame()
)


class PandasDataFrameToPandasSeries(BaseConverter):
    """Convert pd_DataFrame_Table to pd_Series_Table."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "pd_Series_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.DataFrame, store=None) -> pd.Series:
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
    PandasDataFrameToPandasSeries()
)


class ListOfDictToPandasDataFrame(BaseConverter):
    """Convert list_of_dict to pd_DataFrame_Table."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "list_of_dict",
        "mtype_to": "pd_DataFrame_Table",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: list, store=None) -> pd.DataFrame:
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
    ListOfDictToPandasDataFrame()
)


class PandasDataFrameToListOfDict(BaseConverter):
    """Convert pd_DataFrame_Table to list_of_dict."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_DataFrame_Table",
        "mtype_to": "list_of_dict",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.DataFrame, store=None) -> list:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("input is not a pd.DataFrame")

        ret_dict = [obj.loc[i].to_dict() for i in obj.index]

        if isinstance(store, dict):
            store["index"] = obj.index

        return ret_dict


convert_dict[("pd_DataFrame_Table", "list_of_dict", "Table")] = (
    PandasDataFrameToListOfDict()
)

_extend_conversions(
    "pd_Series_Table", "pd_DataFrame_Table", convert_dict, MTYPE_LIST_TABLE
)
_extend_conversions(
    "list_of_dict", "pd_DataFrame_Table", convert_dict, MTYPE_LIST_TABLE
)

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    import polars as pl

    class PandasDataFrameToPolarsEager(BaseConverter):
        """Convert pd_DataFrame_Table to polars_eager_table."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pd_DataFrame_Table",
            "mtype_to": "polars_eager_table",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj: pd.DataFrame, store=None):
            if not isinstance(obj, pd.DataFrame):
                raise TypeError("input is not a pd.DataFrame")

            return pl.DataFrame(obj)

    class PandasDataFrameToPolarsLazy(BaseConverter):
        """Convert pd_DataFrame_Table to polars_lazy_table."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pd_DataFrame_Table",
            "mtype_to": "polars_lazy_table",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj: pd.DataFrame, store=None):
            if not isinstance(obj, pd.DataFrame):
                raise TypeError("input is not a pd.DataFrame")

            return pl.LazyFrame(obj)

    class PolarsEagerToPandasDataFrame(BaseConverter):
        """Convert polars_eager_table to pd_DataFrame_Table."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "polars_eager_table",
            "mtype_to": "pd_DataFrame_Table",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj: pl.DataFrame, store=None):
            if not isinstance(obj, pl.DataFrame):
                raise TypeError("input is not a polars frame")

            return obj.to_pandas()

    class PolarsLazyToPandasDataFrame(BaseConverter):
        """Convert polars_lazy_table to pd_DataFrame_Table."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "polars_lazy_table",
            "mtype_to": "pd_DataFrame_Table",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj: pl.LazyFrame, store=None):
            if not isinstance(obj, pl.LazyFrame):
                raise TypeError("input is not a polars frame")

            return obj.collect().to_pandas()

    class PolarsLazyToPolarsEager(BaseConverter):
        """Convert polars_lazy_table to polars_eager_table."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "polars_lazy_table",
            "mtype_to": "polars_eager_table",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj: pl.LazyFrame, store=None):
            if not isinstance(obj, pl.LazyFrame):
                raise TypeError("input is not a pl.LazyFrame")

            return obj.collect()

    class PolarsEagerToPolarsLazy(BaseConverter):
        """Convert polars_eager_table to polars_lazy_table."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "polars_eager_table",
            "mtype_to": "polars_lazy_table",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj: pl.DataFrame, store=None):
            if not isinstance(obj, pl.DataFrame):
                raise TypeError("input is not a pl.DataFrame")

            return obj.lazy()

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
        PandasDataFrameToPolarsEager()
    )

    convert_dict[("polars_eager_table", "pd_DataFrame_Table", "Table")] = (
        PolarsEagerToPandasDataFrame()
    )
    convert_dict[("polars_lazy_table", "pd_DataFrame_Table", "Table")] = (
        PolarsLazyToPandasDataFrame()
    )

    convert_dict[("pd_DataFrame_Table", "polars_lazy_table", "Table")] = (
        PandasDataFrameToPolarsLazy()
    )

    _extend_conversions(
        "polars_eager_table", "pd_DataFrame_Table", convert_dict, MTYPE_LIST_TABLE
    )
    _extend_conversions(
        "polars_lazy_table", "pd_DataFrame_Table", convert_dict, MTYPE_LIST_TABLE
    )
