"""Example generation for testing.

Exports dict of examples, useful for testing as fixtures.

example_dict: dict indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are data objects, considered examples for the mtype
    all examples with same index are considered "same" on scitype content
    if None, indicates that representation is not possible

example_lossy: dict of bool indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are bool, indicate whether representation has information removed
    all examples with same index are considered "same" on scitype content

example_metadata: dict of metadata dict, indexed by pair
  1st element = considered as this scitype - str
  2nd element = int - index of example
  (there is no "mtype" element, as properties are equal for all mtypes)
elements are metadata dict, as returned by check_is_mtype
    used as expected return of check_is_mtype in tests

overall, conversions from non-lossy representations to any other ones
    should yield the element exactly, identidally (given same index)
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

        return from_pandas(self._get_example("pd.DataFrame", 0), npartitions=1)


class _SeriesUnivPlDataFrame(_SeriesUniv):
    _tags = {
        "mtype": "pl.DataFrame",
        "python_dependencies": "polars>=1.0",
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

        return convert_pandas_to_listDataset(self._get_example("pd.DataFrame", 0))


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
        return pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})


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
        "python_dependencies": "polars>=1.0",
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
        return pd.DataFrame({"a": [1, 4, 0.5, 3], "b": [3, 7, 2, 3 / 7]})


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
        "python_dependencies": "polars>=1.0",
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
        "python_dependencies": "polars>=1.0",
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
