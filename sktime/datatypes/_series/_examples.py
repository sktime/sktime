"""Example generation for testing.

Exports examples of in-memory data containers, useful for testing as fixtures.

Examples come in clusters, tagged by scitype: str, index: int, and metadata: dict.

All examples with the same index are considered "content-wise the same", i.e.,
representing the same abstract data object. They differ by mtype, i.e.,
machine type, which is the specific in-memory representation.

If an example returns None, it indicates that representation
with that specific mtype is not possible.

If the tag "lossy" is True, the representation is considered incomplete,
e.g., metadata such as column names are missing.

Types of tests that can be performed with these examples:

* the mtype and scitype of the example should be correctly inferred by checkers.
* the metadata of hte example should be correctly inferred by checkers.
* conversions from non-lossy representations to any other ones
  should yield the element exactly, identically, for examples of the same index.
"""

import numpy as np
import pandas as pd

from sktime.datatypes._base import BaseExample
from sktime.datatypes._dtypekind import DtypeKind

###
# example 0: univariate


class _SeriesUniv(BaseExample):
    _tags = {
        "scitype": "Series",
        "index": 0,
        "metadata": {
            "is_univariate": True,
            "is_equally_spaced": True,
            "is_empty": False,
            "has_nans": False,
            "n_features": 1,
            "feature_names": ["a"],
            "feature_kind": [DtypeKind.FLOAT],
        },
    }


class _SeriesUnivPdSeries(_SeriesUniv):
    _tags = {
        "mtype": "pd.Series",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.Series([1, 4, 0.5, -3], dtype=np.float64, name="a")


class _SeriesUnivPdDataFrame(_SeriesUniv):
    _tags = {
        "mtype": "pd.DataFrame",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.DataFrame({"a": [1, 4, 0.5, -3]})


class _SeriesUnivNpArray(_SeriesUniv):
    _tags = {
        "mtype": "np.ndarray",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[1], [4], [0.5], [-3]])


class _SeriesUnivXrDataArray(_SeriesUniv):
    _tags = {
        "mtype": "xr.DataArray",
        "python_dependencies": "xarray",
        "lossy": False,
    }

    def build(self):
        import xarray as xr

        return xr.DataArray(
            [[1], [4], [0.5], [-3]],
            coords=[[0, 1, 2, 3], ["a"]],
        )


class _SeriesUnivDaskSeries(_SeriesUniv):
    _tags = {
        "mtype": "dask_series",
        "python_dependencies": "dask",
        "lossy": False,
    }

    def build(self):
        from dask.dataframe import from_pandas

        df = _SeriesUnivPdDataFrame().build()
        return from_pandas(df, npartitions=1)


class _SeriesUnivPlDataFrame(_SeriesUniv):
    _tags = {
        "mtype": "pl.DataFrame",
        "python_dependencies": ["polars>=1.0", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from polars import DataFrame

        return DataFrame(
            {"__index__0": [0, 1, 2, 3], "a": [1, 4, 0.5, -3]}, strict=False
        )


class _SeriesUnivGluontsListDataset(_SeriesUniv):
    _tags = {
        "mtype": "gluonts_ListDataset_series",
        "python_dependencies": "gluonts",
        "lossy": True,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import convert_pandas_to_listDataset

        df = _SeriesUnivPdDataFrame().build()
        return convert_pandas_to_listDataset(df)


###
# example 1: multivariate


class _SeriesMulti(BaseExample):
    _tags = {
        "scitype": "Series",
        "index": 1,
        "metadata": {
            "is_univariate": False,
            "is_equally_spaced": True,
            "is_empty": False,
            "has_nans": False,
            "n_features": 2,
            "feature_names": ["a", "b"],
            "feature_kind": [DtypeKind.FLOAT, DtypeKind.FLOAT],
        },
    }


class _SeriesMultiPdSeries(_SeriesMulti):
    _tags = {
        "mtype": "pd.Series",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return None


class _SeriesMultiPdDataFrame(_SeriesMulti):
    _tags = {
        "mtype": "pd.DataFrame",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})


class _SeriesMultiNpArray(_SeriesMulti):
    _tags = {
        "mtype": "np.ndarray",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]])


class _SeriesMultiXrDataArray(_SeriesMulti):
    _tags = {
        "mtype": "xr.DataArray",
        "python_dependencies": "xarray",
        "lossy": False,
    }

    def build(self):
        import xarray as xr

        return xr.DataArray(
            [[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]],
            coords=[[0, 1, 2, 3], ["a", "b"]],
        )


class _SeriesMultiDaskSeries(_SeriesMulti):
    _tags = {
        "mtype": "dask_series",
        "python_dependencies": "dask",
        "lossy": False,
    }

    def build(self):
        from dask.dataframe import from_pandas

        pd_df = _SeriesMultiPdDataFrame().build()
        return from_pandas(pd_df, npartitions=1)


class _SeriesMultiPlDataFrame(_SeriesMulti):
    _tags = {
        "mtype": "pl.DataFrame",
        "python_dependencies": ["polars>=1.0", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from polars import DataFrame

        return DataFrame(
            {"__index__0": [0, 1, 2, 3], "a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]},
            strict=False,
        )


class _SeriesMultiGluontsListDataset(_SeriesMulti):
    _tags = {
        "mtype": "gluonts_ListDataset_series",
        "python_dependencies": "gluonts",
        "lossy": True,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import convert_pandas_to_listDataset

        pd_df = _SeriesMultiPdDataFrame().build()
        return convert_pandas_to_listDataset(pd_df)


###
# example 2: multivariate, positive


class _SeriesMultiPos(BaseExample):
    _tags = {
        "scitype": "Series",
        "index": 2,
        "metadata": {
            "is_univariate": False,
            "is_equally_spaced": True,
            "is_empty": False,
            "has_nans": False,
            "n_features": 2,
            "feature_names": ["a", "b"],
            "feature_kind": [DtypeKind.FLOAT, DtypeKind.FLOAT],
        },
    }


class _SeriesMultiPosPdSeries(_SeriesMultiPos):
    _tags = {
        "mtype": "pd.Series",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return None


class _SeriesMultiPosPdDataFrame(_SeriesMultiPos):
    _tags = {
        "mtype": "pd.DataFrame",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.DataFrame({"a": [1, 4, 0.5, 3], "b": [3, 7, 2, 3 / 7]})


class _SeriesMultiPosNpArray(_SeriesMultiPos):
    _tags = {
        "mtype": "np.ndarray",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[1, 3], [4, 7], [0.5, 2], [3, 3 / 7]])


class _SeriesMultiPosXrDataArray(_SeriesMultiPos):
    _tags = {
        "mtype": "xr.DataArray",
        "python_dependencies": "xarray",
        "lossy": False,
    }

    def build(self):
        import xarray as xr

        return xr.DataArray(
            [[1, 3], [4, 7], [0.5, 2], [3, 3 / 7]],
            coords=[[0, 1, 2, 3], ["a", "b"]],
        )


class _SeriesMultiPosDaskSeries(_SeriesMultiPos):
    _tags = {
        "mtype": "dask_series",
        "python_dependencies": "dask",
        "lossy": False,
    }

    def build(self):
        from dask.dataframe import from_pandas

        pd_df = _SeriesMultiPosPdDataFrame().build()
        return from_pandas(pd_df, npartitions=1)


class _SeriesMultiPosPlDataFrame(_SeriesMultiPos):
    _tags = {
        "mtype": "pl.DataFrame",
        "python_dependencies": ["polars>=1.0", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from polars import DataFrame

        return DataFrame(
            {"__index__0": [0, 1, 2, 3], "a": [1, 4, 0.5, 3], "b": [3, 7, 2, 3 / 7]},
            strict=False,
        )


class _SeriesMultiPosGluontsListDataset(_SeriesMultiPos):
    _tags = {
        "mtype": "gluonts_ListDataset_series",
        "python_dependencies": "gluonts",
        "lossy": True,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import convert_pandas_to_listDataset

        pd_df = _SeriesMultiPosPdDataFrame().build()
        return convert_pandas_to_listDataset(pd_df)


###
# example 3: univariate, positive


class _SeriesUnivPos(BaseExample):
    _tags = {
        "scitype": "Series",
        "index": 3,
        "metadata": {
            "is_univariate": True,
            "is_equally_spaced": True,
            "is_empty": False,
            "has_nans": False,
            "n_features": 1,
            "feature_names": ["a"],
            "feature_kind": [DtypeKind.FLOAT],
        },
    }


class _SeriesUnivPosPdSeries(_SeriesUnivPos):
    _tags = {
        "mtype": "pd.Series",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.Series([1, 4, 0.5, 3], dtype=np.float64, name="a")


class _SeriesUnivPosPdDataFrame(_SeriesUnivPos):
    _tags = {
        "mtype": "pd.DataFrame",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.DataFrame({"a": [1, 4, 0.5, 3]})


class _SeriesUnivPosNpArray(_SeriesUnivPos):
    _tags = {
        "mtype": "np.ndarray",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[1], [4], [0.5], [3]])


class _SeriesUnivPosXrDataArray(_SeriesUnivPos):
    _tags = {
        "mtype": "xr.DataArray",
        "python_dependencies": "xarray",
        "lossy": False,
    }

    def build(self):
        import xarray as xr

        return xr.DataArray(
            [[1], [4], [0.5], [3]],
            coords=[[0, 1, 2, 3], ["a"]],
        )


class _SeriesUnivPosPlDataFrame(_SeriesUnivPos):
    _tags = {
        "mtype": "pl.DataFrame",
        "python_dependencies": ["polars>=1.0", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from polars import DataFrame

        return DataFrame(
            {"__index__0": [0, 1, 2, 3], "a": [1, 4, 0.5, 3]}, strict=False
        )


class _SeriesUnivPosGluontsListDataset(_SeriesUnivPos):
    _tags = {
        "mtype": "gluonts_ListDataset_series",
        "python_dependencies": "gluonts",
        "lossy": True,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import convert_pandas_to_listDataset

        pd_df = _SeriesUnivPosPdDataFrame().build()
        return convert_pandas_to_listDataset(pd_df)
