"""Machine type converters for Series scitype.

Exports conversion and mtype dictionary for Series scitype:

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

__author__ = ["fkiraly"]

__all__ = ["convert_dict"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datatypes._base import BaseConverter

##############################################################
# methods to convert one machine type to another machine type
##############################################################
from sktime.datatypes._convert_utils._coerce import _coerce_df_dtypes
from sktime.datatypes._convert_utils._convert import _extend_conversions

# this needs to be refactored with the convert module
MTYPE_LIST_SERIES = [
    "pd.Series",
    "pd.DataFrame",
    "np.ndarray",
    "xr.DataArray",
    "dask_series",
    "pl.DataFrame",
    "gluonts_ListDataset_series",
    "gluonts_PandasDataset_series",
]


class SeriesIdentity(BaseConverter):
    """Identity converter for Series mtypes."""

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
        return [(tp, tp) for tp in MTYPE_LIST_SERIES]

    def _convert(self, obj, store=None):
        obj = _coerce_df_dtypes(obj)
        return obj


convert_dict = dict()


class SeriesToDataFrame(BaseConverter):
    """Convert pd.Series to pd.DataFrame."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd.Series",
        "mtype_to": "pd.DataFrame",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.Series, store=None) -> pd.DataFrame:
        if not isinstance(obj, pd.Series):
            raise TypeError("input must be a pd.Series")

        obj = _coerce_df_dtypes(obj)

        if isinstance(store, dict):
            store["name"] = obj.name

        res = pd.DataFrame(obj)

        if (
            isinstance(store, dict)
            and "columns" in store.keys()
            and len(store["columns"]) == 1
        ):
            res.columns = store["columns"]

        return res


class DataFrameToSeries(BaseConverter):
    """Convert pd.DataFrame to pd.Series."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd.DataFrame",
        "mtype_to": "pd.Series",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.DataFrame, store=None) -> pd.Series:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("input is not a pd.DataFrame")

        obj = _coerce_df_dtypes(obj)

        if len(obj.columns) != 1:
            raise ValueError("input must be univariate pd.DataFrame, with one column")

        if isinstance(store, dict):
            store["columns"] = obj.columns[[0]]

        y = obj[obj.columns[0]]

        if isinstance(store, dict) and "name" in store.keys():
            # column name becomes attr name
            y.name = store["name"]
        else:
            y.name = None

        return y


def convert_MvS_to_UvS_as_Series(obj: pd.DataFrame, store=None) -> pd.Series:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("input is not a pd.DataFrame")

    obj = _coerce_df_dtypes(obj)

    if len(obj.columns) != 1:
        raise ValueError("input must be univariate pd.DataFrame, with one column")

    if isinstance(store, dict):
        store["columns"] = obj.columns[[0]]

    y = obj[obj.columns[0]]

    if isinstance(store, dict) and "name" in store.keys():
        # column name becomes attr name
        y.name = store["name"]
    else:
        y.name = None

    return y


class DataFrameToNumpy(BaseConverter):
    """Convert pd.DataFrame to np.ndarray."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd.DataFrame",
        "mtype_to": "np.ndarray",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.DataFrame, store=None) -> np.ndarray:
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("input must be a pd.DataFrame")

        obj = _coerce_df_dtypes(obj)

        if isinstance(store, dict):
            store["columns"] = obj.columns
            store["index"] = obj.index

        return obj.to_numpy(dtype="float")


class SeriesToNumpy(BaseConverter):
    """Convert pd.Series to np.ndarray."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd.Series",
        "mtype_to": "np.ndarray",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: pd.Series, store=None) -> np.ndarray:
        if not isinstance(obj, pd.Series):
            raise TypeError("input must be a pd.Series")

        obj = _coerce_df_dtypes(obj)

        if isinstance(store, dict):
            store["index"] = obj.index
            store["name"] = obj.name

        return pd.DataFrame(obj).to_numpy(dtype="float")


class NumpyToDataFrame(BaseConverter):
    """Convert np.ndarray to pd.DataFrame."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "np.ndarray",
        "mtype_to": "pd.DataFrame",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: np.ndarray, store=None) -> pd.DataFrame:
        if not isinstance(obj, np.ndarray) and len(obj.shape) > 2:
            raise TypeError("input must be a np.ndarray of dim 1 or 2")

        if len(obj.shape) == 1:
            obj = np.reshape(obj, (-1, 1))

        res = pd.DataFrame(obj)

        # add column names or index from store if stored and length fits
        if (
            isinstance(store, dict)
            and "columns" in store.keys()
            and len(store["columns"]) == obj.shape[1]
        ):
            res.columns = store["columns"]
        if (
            isinstance(store, dict)
            and "index" in store.keys()
            and len(store["index"]) == obj.shape[0]
        ):
            res.index = store["index"]

        return res


class NumpyToSeries(BaseConverter):
    """Convert np.ndarray to pd.Series."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "np.ndarray",
        "mtype_to": "pd.Series",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": None,
    }

    def _convert(self, obj: np.ndarray, store=None) -> pd.Series:
        if not isinstance(obj, np.ndarray) or obj.ndim > 2:
            raise TypeError("input must be a one-column np.ndarray of dim 1 or 2")

        if obj.ndim == 2 and obj.shape[1] != 1:
            raise TypeError("input must be a one-column np.ndarray of dim 1 or 2")

        res = pd.Series(obj.flatten())

        # add index from store if stored and length fits
        if (
            isinstance(store, dict)
            and "index" in store.keys()
            and len(store["index"]) == obj.shape[0]
        ):
            res.index = store["index"]

        if isinstance(store, dict) and "name" in store.keys():
            res.name = store["name"]

        return res


if _check_soft_dependencies("xarray", severity="none"):
    import xarray as xr

    class XrDataArrayToDataFrame(BaseConverter):
        """Convert xr.DataArray to pd.DataFrame."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "xr.DataArray",
            "mtype_to": "pd.DataFrame",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj: xr.DataArray, store=None) -> pd.DataFrame:
            if not isinstance(obj, xr.DataArray):
                raise TypeError("input must be a xr.DataArray")

            if isinstance(store, dict):
                store["coords"] = list(obj.coords.keys())

            index = obj.indexes[obj.dims[0]]
            columns = obj.indexes[obj.dims[1]] if len(obj.dims) == 2 else None
            df = pd.DataFrame(obj.values, index=index, columns=columns)

            # int64 coercions are needed due to inconsistencies specifically on windows
            df = df.astype(
                dict.fromkeys(df.select_dtypes(include="int32").columns, "int64")
            )

            if df.index.dtype == "int32":
                df.index = df.index.astype("int64")

            return df

    class DataFrameToXrDataArray(BaseConverter):
        """Convert pd.DataFrame to xr.DataArray."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pd.DataFrame",
            "mtype_to": "xr.DataArray",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj: pd.DataFrame, store=None) -> xr.DataArray:
            if not isinstance(obj, pd.DataFrame):
                raise TypeError("input must be a xr.DataArray")

            obj = _coerce_df_dtypes(obj)

            result = xr.DataArray(obj.values, coords=[obj.index, obj.columns])

            if isinstance(store, dict) and "coords" in store:
                result = result.rename(
                    dict(zip(list(result.coords.keys()), store["coords"]))
                )

            return result

    convert_dict[("xr.DataArray", "pd.DataFrame", "Series")] = XrDataArrayToDataFrame()

    convert_dict[("pd.DataFrame", "xr.DataArray", "Series")] = DataFrameToXrDataArray()

    _extend_conversions(
        "xr.DataArray", "pd.DataFrame", convert_dict, mtype_universe=MTYPE_LIST_SERIES
    )


if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import (
        convert_dask_to_pandas,
        convert_pandas_to_dask,
    )

    class DaskToDataFrame(BaseConverter):
        """Convert dask_series to pd.DataFrame."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "dask_series",
            "mtype_to": "pd.DataFrame",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None) -> pd.DataFrame:
            return convert_dask_to_pandas(obj)

    class DataFrameToDask(BaseConverter):
        """Convert pd.DataFrame to dask_series."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pd.DataFrame",
            "mtype_to": "dask_series",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None):
            return convert_pandas_to_dask(obj)

    convert_dict[("dask_series", "pd.DataFrame", "Series")] = DaskToDataFrame()

    convert_dict[("pd.DataFrame", "dask_series", "Series")] = DataFrameToDask()

    _extend_conversions(
        "dask_series", "pd.DataFrame", convert_dict, mtype_universe=MTYPE_LIST_SERIES
    )


if _check_soft_dependencies("polars", severity="none"):
    from sktime.datatypes._adapter.polars import (
        convert_pandas_to_polars,
        convert_polars_to_pandas,
    )

    class PolarsToSeries(BaseConverter):
        """Convert pl.DataFrame to pd.Series."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pl.DataFrame",
            "mtype_to": "pd.Series",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None) -> pd.Series:
            pd_df = convert_polars_to_pandas(obj)
            return convert_MvS_to_UvS_as_Series(pd_df, store=store)

    class PolarsToDataFrame(BaseConverter):
        """Convert pl.DataFrame to pd.DataFrame."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pl.DataFrame",
            "mtype_to": "pd.DataFrame",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None) -> pd.DataFrame:
            return convert_polars_to_pandas(obj)

    class DataFrameToPolars(BaseConverter):
        """Convert a pandas DataFrame to a polars DataFrame."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pd.DataFrame",
            "mtype_to": "pl.DataFrame",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None):
            return convert_pandas_to_polars(obj)

    class SeriesToPolars(BaseConverter):
        """Convert a pandas Series to a polars DataFrame."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pd.Series",
            "mtype_to": "pl.DataFrame",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None):
            return convert_pandas_to_polars(obj)

    def convert_polars_lazy_to_mvs_as_series(obj, store=None):
        return convert_polars_to_pandas(obj)

    convert_dict[("pl.LazyFrame", "pd.DataFrame", "Series")] = (
        convert_polars_lazy_to_mvs_as_series
    )

    def convert_mvs_to_polars_lazy_as_series(obj, store=None):
        return convert_pandas_to_polars(obj, lazy=True)

    convert_dict[("pd.DataFrame", "pl.LazyFrame", "Series")] = (
        convert_mvs_to_polars_lazy_as_series
    )


if _check_soft_dependencies("gluonts", severity="none"):
    from sktime.datatypes._adapter.gluonts import (
        convert_listDataset_to_pandas,
        convert_pandas_dataframe_to_pandasDataset,
        convert_pandas_to_listDataset,
        convert_pandasDataset_to_pandas_dataframe,
    )

    class GluontsListDatasetToDataFrame(BaseConverter):
        """Convert a GluonTS ListDataset to a pandas DataFrame."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "gluonts_ListDataset_series",
            "mtype_to": "pd.DataFrame",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None):
            return convert_listDataset_to_pandas(obj)

    class DataFrameToGluontsListDataset(BaseConverter):
        """Convert a pandas DataFrame to a GluonTS ListDataset."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pd.DataFrame",
            "mtype_to": "gluonts_ListDataset_series",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None):
            return convert_pandas_to_listDataset(obj)

    class GluontsPandasDatasetToDataFrame(BaseConverter):
        """Convert a GluonTS PandasDataset to a pandas DataFrame."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "gluonts_PandasDataset_series",
            "mtype_to": "pd.DataFrame",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None):
            return convert_pandasDataset_to_pandas_dataframe(obj)

    class DataFrameToGluontsPandasDataset(BaseConverter):
        """Convert a pandas DataFrame to a GluonTS PandasDataset."""

        _tags = {
            "object_type": "converter",
            "mtype_from": "pd.DataFrame",
            "mtype_to": "gluonts_PandasDataset_series",
            "multiple_conversions": False,
            "python_version": None,
            "python_dependencies": None,
        }

        def _convert(self, obj, store=None):
            return convert_pandas_dataframe_to_pandasDataset(obj)

    # Storing functions in convert_dict
    convert_dict[("pd.DataFrame", "gluonts_ListDataset_series", "Series")] = (
        DataFrameToGluontsListDataset()
    )

    convert_dict[("gluonts_ListDataset_series", "pd.DataFrame", "Series")] = (
        GluontsListDatasetToDataFrame()
    )

    convert_dict[("pd.DataFrame", "gluonts_PandasDataset_series", "Series")] = (
        DataFrameToGluontsPandasDataset()
    )

    convert_dict[("gluonts_PandasDataset_series", "pd.DataFrame", "Series")] = (
        GluontsPandasDatasetToDataFrame()
    )

    # Extending conversions
    _extend_conversions(
        "gluonts_ListDataset_series",
        "pd.DataFrame",
        convert_dict,
        mtype_universe=MTYPE_LIST_SERIES,
    )

    _extend_conversions(
        "gluonts_PandasDataset_series",
        "pd.DataFrame",
        convert_dict,
        mtype_universe=MTYPE_LIST_SERIES,
    )
