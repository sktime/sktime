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


class _UnivTable(BaseExample):
    _tags = {
        "scitype": "Table",
        "index": 0,
        "metadata": {
            "is_univariate": True,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 4,
            "n_features": 1,
            "feature_names": ["a"],
            "feature_kind": [DtypeKind.FLOAT],
        },
    }


class _UnivTableDf(_UnivTable):
    _tags = {
        "mtype": "pd_DataFrame_Table",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.DataFrame({"a": [1, 4, 0.5, -3]})


class _UnivTableNumpy2D(_UnivTable):
    _tags = {
        "mtype": "numpy2D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[1], [4], [0.5], [-3]])


class _UnivTableNumpy1D(_UnivTable):
    _tags = {
        "mtype": "numpy1D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([1, 4, 0.5, -3])


class _UnivTableSeries(_UnivTable):
    _tags = {
        "mtype": "pd_Series_Table",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return pd.Series([1, 4, 0.5, -3])


class _UnivTableListOfDict(_UnivTable):
    _tags = {
        "mtype": "list_of_dict",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return [{"a": 1.0}, {"a": 4.0}, {"a": 0.5}, {"a": -3.0}]


class _UnivTablePolarsEager(_UnivTable):
    _tags = {
        "mtype": "polars_eager_table",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        import polars as pl

        df = pd.DataFrame({"a": [1, 4, 0.5, -3]})
        return pl.DataFrame(df)


class _UnivTablePolarsLazy(_UnivTable):
    _tags = {
        "mtype": "polars_lazy_table",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        import polars as pl

        df = pd.DataFrame({"a": [1, 4, 0.5, -3]})
        return pl.LazyFrame(df)


###
# example 1: multivariate


class _MultivTable(BaseExample):
    _tags = {
        "scitype": "Table",
        "index": 1,
        "metadata": {
            "is_univariate": False,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 4,
            "n_features": 2,
            "feature_names": ["a", "b"],
            "feature_kind": [DtypeKind.FLOAT, DtypeKind.FLOAT],
        },
    }


class _MultivTableDf(_MultivTable):
    _tags = {
        "mtype": "pd_DataFrame_Table",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})


class _MultivTableNumpy2D(_MultivTable):
    _tags = {
        "mtype": "numpy2D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]])


class _MultivTableNumpy1D(_MultivTable):
    _tags = {
        "mtype": "numpy1D",
        "python_dependencies": None,
        "lossy": None,
    }

    def build(self):
        return None


class _MultivTableSeries(_MultivTable):
    _tags = {
        "mtype": "pd_Series_Table",
        "python_dependencies": None,
        "lossy": None,
    }

    def build(self):
        return None


class _MultivTableListOfDict(_MultivTable):
    _tags = {
        "mtype": "list_of_dict",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return [
            {"a": 1.0, "b": 3.0},
            {"a": 4.0, "b": 7.0},
            {"a": 0.5, "b": 2.0},
            {"a": -3.0, "b": -3 / 7},
        ]


class _MultivTablePolarsEager(_MultivTable):
    _tags = {
        "mtype": "polars_eager_table",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        import polars as pl

        df = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})
        return pl.DataFrame(df)


class _MultivTablePolarsLazy(_MultivTable):
    _tags = {
        "mtype": "polars_lazy_table",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        import polars as pl

        df = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})
        return pl.LazyFrame(df)
