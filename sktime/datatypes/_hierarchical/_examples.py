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
* the metadata of the example should be correctly inferred by checkers.
* conversions from non-lossy representations to any other ones
  should yield the element exactly, identically, for examples of the same index.
"""

import pandas as pd

from sktime.datatypes._base import BaseExample
from sktime.datatypes._dtypekind import DtypeKind

###
# example 0: multivariate, equally sampled


class _HierMultivEqspl(BaseExample):
    _tags = {
        "scitype": "Hierarchical",
        "index": 0,
        "metadata": {
            "is_univariate": False,
            "is_one_panel": False,
            "is_one_series": False,
            "is_equally_spaced": True,
            "is_equal_length": True,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 6,
            "n_panels": 2,
            "n_features": 2,
            "feature_names": ["var_0", "var_1"],
            "feature_kind": [DtypeKind.FLOAT, DtypeKind.FLOAT],
        },
    }


class _HierMultivEqsplPdMiHier(_HierMultivEqspl):
    _tags = {
        "mtype": "pd_multiindex_hier",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        cols = ["foo", "bar", "timepoints"] + [f"var_{i}" for i in range(2)]

        Xlist = [
            pd.DataFrame(
                [["a", 0, 0, 1, 4], ["a", 0, 1, 2, 5], ["a", 0, 2, 3, 6]], columns=cols
            ),
            pd.DataFrame(
                [["a", 1, 0, 1, 4], ["a", 1, 1, 2, 55], ["a", 1, 2, 3, 6]], columns=cols
            ),
            pd.DataFrame(
                [["a", 2, 0, 1, 42], ["a", 2, 1, 2, 5], ["a", 2, 2, 3, 6]], columns=cols
            ),
            pd.DataFrame(
                [["b", 0, 0, 1, 4], ["b", 0, 1, 2, 5], ["b", 0, 2, 3, 6]], columns=cols
            ),
            pd.DataFrame(
                [["b", 1, 0, 1, 4], ["b", 1, 1, 2, 55], ["b", 1, 2, 3, 6]], columns=cols
            ),
            pd.DataFrame(
                [["b", 2, 0, 1, 42], ["b", 2, 1, 2, 5], ["b", 2, 2, 3, 6]], columns=cols
            ),
        ]

        X = pd.concat(Xlist)
        X = X.set_index(["foo", "bar", "timepoints"])
        return X


class _HierMultivEqsplDaskHier(_HierMultivEqspl):
    _tags = {
        "mtype": "dask_hierarchical",
        "python_dependencies": ["dask"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

        X = _HierMultivEqsplPdMiHier().build()
        return convert_pandas_to_dask(X, npartitions=1)


class _HierMultivEqsplPolarsHier(_HierMultivEqsplPdMiHier):
    _tags = {
        "mtype": "polars_hierarchical",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.polars import convert_pandas_to_polars

        X_mi = _HierMultivEqsplPdMiHier().build()
        pl_frame = convert_pandas_to_polars(X_mi)
        return pl_frame


###
# example 1: univariate, equally sampled


class _HierUnivEqspl(BaseExample):
    _tags = {
        "scitype": "Hierarchical",
        "index": 1,
        "metadata": {
            "is_univariate": True,
            "is_one_panel": False,
            "is_one_series": False,
            "is_equally_spaced": True,
            "is_equal_length": True,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 6,
            "n_panels": 2,
            "n_features": 1,
            "feature_names": ["var_0"],
            "feature_kind": [DtypeKind.FLOAT],
        },
    }


class _HierUnivEqsplPdMiHier(_HierUnivEqspl):
    _tags = {
        "mtype": "pd_multiindex_hier",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        col = ["foo", "bar", "timepoints"] + [f"var_{i}" for i in range(1)]

        Xlist = [
            pd.DataFrame([["a", 0, 0, 1], ["a", 0, 1, 2], ["a", 0, 2, 3]], columns=col),
            pd.DataFrame([["a", 1, 0, 1], ["a", 1, 1, 2], ["a", 1, 2, 3]], columns=col),
            pd.DataFrame([["a", 2, 0, 1], ["a", 2, 1, 2], ["a", 2, 2, 3]], columns=col),
            pd.DataFrame([["b", 0, 0, 1], ["b", 0, 1, 2], ["b", 0, 2, 3]], columns=col),
            pd.DataFrame([["b", 1, 0, 1], ["b", 1, 1, 2], ["b", 1, 2, 3]], columns=col),
            pd.DataFrame([["b", 2, 0, 1], ["b", 2, 1, 2], ["b", 2, 2, 3]], columns=col),
        ]
        X = pd.concat(Xlist)
        X = X.set_index(["foo", "bar", "timepoints"])
        return X


class _HierUnivEqsplDaskHier(_HierUnivEqspl):
    _tags = {
        "mtype": "dask_hierarchical",
        "python_dependencies": ["dask"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

        X = _HierUnivEqsplPdMiHier().build()
        return convert_pandas_to_dask(X, npartitions=1)


class _HierUnivEqsplPolarsHier(_HierUnivEqsplPdMiHier):
    _tags = {
        "mtype": "polars_hierarchical",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from sktime.datatypes._adapter.polars import convert_pandas_to_polars

        X_mi = _HierUnivEqsplPdMiHier().build()
        pl_frame = convert_pandas_to_polars(X_mi)
        return pl_frame
