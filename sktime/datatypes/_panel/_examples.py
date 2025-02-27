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
# example 0: multivariate, equally sampled


class _PanelMultivEqspl(BaseExample):
    _tags = {
        "scitype": "Panel",
        "index": 0,
        "metadata": {
            "is_univariate": False,
            "is_one_series": False,
            "n_panels": 1,
            "is_one_panel": True,
            "is_equally_spaced": True,
            "is_equal_length": True,
            "is_equal_index": True,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 3,
            "n_features": 2,
            "feature_names": ["var_0", "var_1"],
            "feature_kind": [DtypeKind.FLOAT, DtypeKind.FLOAT],
        },
    }


class _PanelMultivEqsplNumpy3D(_PanelMultivEqspl):
    _tags = {
        "mtype": "numpy3D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        X = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 55, 6]], [[1, 2, 3], [42, 5, 6]]],
            dtype=np.int64,
        )
        return X


class _PanelMultivEqsplNumpyFlat(_PanelMultivEqspl):
    _tags = {
        "mtype": "numpyflat",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return None


class _PanelMultivEqsplDfList(_PanelMultivEqspl):
    _tags = {
        "mtype": "df-list",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = [f"var_{i}" for i in range(2)]
        Xlist = [
            pd.DataFrame([[1, 4], [2, 5], [3, 6]], columns=cols),
            pd.DataFrame([[1, 4], [2, 55], [3, 6]], columns=cols),
            pd.DataFrame([[1, 42], [2, 5], [3, 6]], columns=cols),
        ]
        return Xlist


class _PanelMultivEqsplPdMultiindex(_PanelMultivEqspl):
    _tags = {
        "mtype": "pd-multiindex",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = ["instances", "timepoints"] + [f"var_{i}" for i in range(2)]
        Xlist = [
            pd.DataFrame([[0, 0, 1, 4], [0, 1, 2, 5], [0, 2, 3, 6]], columns=cols),
            pd.DataFrame([[1, 0, 1, 4], [1, 1, 2, 55], [1, 2, 3, 6]], columns=cols),
            pd.DataFrame([[2, 0, 1, 42], [2, 1, 2, 5], [2, 2, 3, 6]], columns=cols),
        ]
        X = pd.concat(Xlist)
        X = X.set_index(["instances", "timepoints"])
        return X


class _PanelMultivEqsplNestedUniv(_PanelMultivEqspl):
    _tags = {
        "mtype": "nested_univ",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = [f"var_{i}" for i in range(2)]
        X = pd.DataFrame(columns=cols, index=pd.RangeIndex(3))
        nestes_series = [
            pd.Series([1, 2, 3], index=pd.Index([0, 1, 2], name="timepoints")),
            pd.Series([1, 2, 3], index=pd.Index([0, 1, 2], name="timepoints")),
            pd.Series([1, 2, 3], index=pd.Index([0, 1, 2], name="timepoints")),
        ]
        X["var_0"] = pd.Series(nestes_series)
        nestes_series_2 = [
            pd.Series([4, 5, 6], index=pd.Index([0, 1, 2], name="timepoints")),
            pd.Series([4, 55, 6], index=pd.Index([0, 1, 2], name="timepoints")),
            pd.Series([42, 5, 6], index=pd.Index([0, 1, 2], name="timepoints")),
        ]
        X["var_1"] = pd.Series(nestes_series_2)
        X.index.name = "instances"
        return X


class _PanelMultivEqsplDaskPanel(_PanelMultivEqspl):
    _tags = {
        "mtype": "dask_panel",
        "python_dependencies": ["dask"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

        pd_df = _PanelMultivEqsplPdMultiindex().build()
        df_dask = convert_pandas_to_dask(pd_df, npartitions=1)
        return df_dask


class _PanelMultivEqsplPolarsPanel(_PanelMultivEqspl):
    _tags = {
        "mtype": "polars_panel",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.polars import convert_pandas_to_polars

        pd_df = _PanelMultivEqsplPdMultiindex().build()
        pl_frame = convert_pandas_to_polars(pd_df)
        return pl_frame


class _PanelMultivEqsplGluontsListDatasetPanel(_PanelMultivEqspl):
    _tags = {
        "mtype": "gluonts_ListDataset_panel",
        "python_dependencies": ["gluonts"],
        "lossy": True,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import convert_pandas_to_listDataset

        pd_df = _PanelMultivEqsplPdMultiindex().build()
        list_dataset = convert_pandas_to_listDataset(pd_df)
        return list_dataset


class _PanelMultivEqsplGluontsPandasDatasetPanel(_PanelMultivEqspl):
    _tags = {
        "mtype": "gluonts_PandasDataset_panel",
        "python_dependencies": ["gluonts"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import (
            convert_pandas_multiindex_to_pandasDataset,
        )

        pd_df = _PanelMultivEqsplPdMultiindex().build()
        pandas_dataset = convert_pandas_multiindex_to_pandasDataset(
            pd_df,
            item_id="instances",
            timepoints="timepoints",
            target=["var_0", "var_1"],
        )
        return pandas_dataset


###
# example 1: univariate, equally sampled


class _PanelUnivEqspl(BaseExample):
    _tags = {
        "scitype": "Panel",
        "index": 1,
        "metadata": {
            "is_univariate": True,
            "is_one_series": False,
            "n_panels": 1,
            "is_one_panel": True,
            "is_equally_spaced": True,
            "is_equal_length": True,
            "is_equal_index": True,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 3,
            "n_features": 1,
            "feature_names": ["var_0"],
            "feature_kind": [DtypeKind.FLOAT],
        },
    }


class _PanelUnivEqsplNumpy3D(_PanelUnivEqspl):
    _tags = {
        "mtype": "numpy3D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[[4, 5, 6]], [[4, 55, 6]], [[42, 5, 6]]], dtype=np.int64)


class _PanelUnivEqsplNumpyFlat(_PanelUnivEqspl):
    _tags = {
        "mtype": "numpyflat",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[4, 5, 6], [4, 55, 6], [42, 5, 6]], dtype=np.int64)


class _PanelUnivEqsplDfList(_PanelUnivEqspl):
    _tags = {
        "mtype": "df-list",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = [f"var_{i}" for i in range(1)]
        Xlist = [
            pd.DataFrame([[4], [5], [6]], columns=cols),
            pd.DataFrame([[4], [55], [6]], columns=cols),
            pd.DataFrame([[42], [5], [6]], columns=cols),
        ]
        return Xlist


class _PanelUnivEqsplPdMultiindex(_PanelUnivEqspl):
    _tags = {
        "mtype": "pd-multiindex",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = ["instances", "timepoints"] + [f"var_{i}" for i in range(1)]
        Xlist = [
            pd.DataFrame([[0, 0, 4], [0, 1, 5], [0, 2, 6]], columns=cols),
            pd.DataFrame([[1, 0, 4], [1, 1, 55], [1, 2, 6]], columns=cols),
            pd.DataFrame([[2, 0, 42], [2, 1, 5], [2, 2, 6]], columns=cols),
        ]
        X = pd.concat(Xlist)
        X = X.set_index(["instances", "timepoints"])
        return X


class _PanelUnivEqsplNestedUniv(_PanelUnivEqspl):
    _tags = {
        "mtype": "nested_univ",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = [f"var_{i}" for i in range(1)]
        X = pd.DataFrame(columns=cols, index=pd.RangeIndex(3))
        data = [
            pd.Series([4, 5, 6], index=pd.Index([0, 1, 2], name="timepoints")),
            pd.Series([4, 55, 6], index=pd.Index([0, 1, 2], name="timepoints")),
            pd.Series([42, 5, 6], index=pd.Index([0, 1, 2], name="timepoints")),
        ]
        X["var_0"] = pd.Series(data)
        X.index.name = "instances"
        return X


class _PanelUnivEqsplDaskPanel(_PanelUnivEqspl):
    _tags = {
        "mtype": "dask_panel",
        "python_dependencies": ["dask"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

        pd_df = _PanelUnivEqsplPdMultiindex().build()
        df_dask = convert_pandas_to_dask(pd_df, npartitions=1)
        return df_dask


class _PanelUnivEqsplPolarsPanel(_PanelUnivEqspl):
    _tags = {
        "mtype": "polars_panel",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.polars import convert_pandas_to_polars

        pd_df = _PanelUnivEqsplPdMultiindex().build()
        pl_frame = convert_pandas_to_polars(pd_df)
        return pl_frame


class _PanelUnivEqsplGluontsListDatasetPanel(_PanelUnivEqspl):
    _tags = {
        "mtype": "gluonts_ListDataset_panel",
        "python_dependencies": ["gluonts"],
        "lossy": True,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import convert_pandas_to_listDataset

        pd_df = _PanelUnivEqsplPdMultiindex().build()
        list_dataset = convert_pandas_to_listDataset(pd_df)
        return list_dataset


class _PanelUnivEqsplGluontsPandasDatasetPanel(_PanelUnivEqspl):
    _tags = {
        "mtype": "gluonts_PandasDataset_panel",
        "python_dependencies": ["gluonts"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import (
            convert_pandas_multiindex_to_pandasDataset,
        )

        pd_df = _PanelUnivEqsplPdMultiindex().build()
        pandas_dataset = convert_pandas_multiindex_to_pandasDataset(
            pd_df, item_id="instances", timepoints="timepoints", target=["var_0"]
        )
        return pandas_dataset


###
# example 2: univariate, equally sampled, one series


class _PanelUnivEqsplOneSeries(BaseExample):
    _tags = {
        "scitype": "Panel",
        "index": 2,
        "metadata": {
            "is_univariate": True,
            "is_one_series": True,
            "n_panels": 1,
            "is_one_panel": True,
            "is_equally_spaced": True,
            "is_equal_length": True,
            "is_equal_index": True,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 1,
            "n_features": 1,
            "feature_names": ["var_0"],
            "feature_kind": [DtypeKind.FLOAT],
        },
    }


class _PanelUnivEqsplOneSeriesNumpy3D(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "numpy3D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[[4, 5, 6]]], dtype=np.int64)


class _PanelUnivEqsplOneSeriesNumpyFlat(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "numpyflat",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[4, 5, 6]], dtype=np.int64)


class _PanelUnivEqsplOneSeriesDfList(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "df-list",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = [f"var_{i}" for i in range(1)]
        Xlist = [pd.DataFrame([[4], [5], [6]], columns=cols)]
        return Xlist


class _PanelUnivEqsplOneSeriesPdMultiindex(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "pd-multiindex",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = ["instances", "timepoints"] + [f"var_{i}" for i in range(1)]
        Xlist = [pd.DataFrame([[0, 0, 4], [0, 1, 5], [0, 2, 6]], columns=cols)]
        X = pd.concat(Xlist)
        X = X.set_index(["instances", "timepoints"])
        return X


class _PanelUnivEqsplOneSeriesNestedUniv(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "nested_univ",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = [f"var_{i}" for i in range(1)]
        X = pd.DataFrame(columns=cols, index=pd.RangeIndex(1))
        X["var_0"] = pd.Series(
            [pd.Series([4, 5, 6], index=pd.Index([0, 1, 2], name="timepoints"))]
        )
        X.index.name = "instances"
        return X


class _PanelUnivEqsplOneSeriesDaskPanel(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "dask_panel",
        "python_dependencies": ["dask"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

        pd_df = _PanelUnivEqsplOneSeriesPdMultiindex().build()
        df_dask = convert_pandas_to_dask(pd_df, npartitions=1)
        return df_dask


class _PanelUnivEqsplOneSeriesPolarsPanel(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "polars_panel",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.polars import convert_pandas_to_polars

        pd_df = _PanelUnivEqsplOneSeriesPdMultiindex().build()
        pl_frame = convert_pandas_to_polars(pd_df)
        return pl_frame


class _PanelUnivEqsplOneSeriesGluontsListDatasetPanel(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "gluonts_ListDataset_panel",
        "python_dependencies": ["gluonts"],
        "lossy": True,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import convert_pandas_to_listDataset

        pd_df = _PanelUnivEqsplOneSeriesPdMultiindex().build()
        list_dataset = convert_pandas_to_listDataset(pd_df)
        return list_dataset


class _PanelUnivEqsplOneSeriesGluontsPandasDatasetPanel(_PanelUnivEqsplOneSeries):
    _tags = {
        "mtype": "gluonts_PandasDataset_panel",
        "python_dependencies": ["gluonts"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.gluonts import (
            convert_pandas_multiindex_to_pandasDataset,
        )

        pd_df = _PanelUnivEqsplOneSeriesPdMultiindex().build()
        pandas_dataset = convert_pandas_multiindex_to_pandasDataset(
            pd_df, item_id="instances", timepoints="timepoints", target=["var_0"]
        )
        return pandas_dataset


###
# example 3: univariate, equally sampled, lossy,
# targets #4299 pd-multiindex panel incorrect is_equally_spaced


class _PanelUnivEqsplLossy(BaseExample):
    _tags = {
        "scitype": "Panel",
        "index": 3,
        "metadata": {
            "is_univariate": True,
            "is_one_series": False,
            "n_panels": 1,
            "is_one_panel": True,
            "is_equally_spaced": True,
            "is_equal_length": True,
            "is_equal_index": False,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 3,
            "n_features": 1,
            "feature_names": ["var_0"],
            "feature_kind": [DtypeKind.FLOAT],
        },
    }


class _PanelUnivEqsplLossyPdMultiindex(_PanelUnivEqsplLossy):
    _tags = {
        "mtype": "pd-multiindex",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        X_instances = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        X_timepoints = pd.to_datetime([0, 1, 2, 4, 5, 6, 9, 10, 11], unit="s")
        X_multiindex = pd.MultiIndex.from_arrays(
            [X_instances, X_timepoints], names=["instances", "timepoints"]
        )

        X = pd.DataFrame(index=X_multiindex, data=list(range(0, 9)), columns=["var_0"])
        return X
