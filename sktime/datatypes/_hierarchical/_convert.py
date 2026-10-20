"""Machine type converters for Hierarchical scitype.

Conversions are implemented as ``BaseConverter`` classes, one class per
conversion, and collected by ``generate_convert_dict`` in ``datatypes._convert``.
"""

__author__ = ["fkiraly"]

from sktime.datatypes._base import BaseConverter
from sktime.datatypes._convert_utils._coerce import _coerce_df_dtypes

# this needs to be refactored with the convert module
MTYPE_LIST_HIERARCHICAL = [
    "pd_multiindex_hier",
    "dask_hierarchical",
    "polars_hierarchical",
]


class HierarchicalIdentity(BaseConverter):
    """All Hierarchical scitype conversions of any mtype to itself.

    Coerces pandas nullable dtypes, does nothing if obj is not pandas.
    """

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
        return [(tp, tp) for tp in MTYPE_LIST_HIERARCHICAL]

    def _convert(self, obj, store=None):
        return _coerce_df_dtypes(obj)


class HierarchicalDaskToPdMultiIndex(BaseConverter):
    """Conversion: dask_hierarchical -> pd_multiindex_hier, of Hierarchical scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "dask_hierarchical",
        "mtype_to": "pd_multiindex_hier",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": "dask",
    }

    def _convert(self, obj, store=None):
        from sktime.datatypes._adapter.dask_to_pd import convert_dask_to_pandas

        return convert_dask_to_pandas(obj)


class HierarchicalPdMultiIndexToDask(BaseConverter):
    """Conversion: pd_multiindex_hier -> dask_hierarchical, of Hierarchical scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_multiindex_hier",
        "mtype_to": "dask_hierarchical",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": "dask",
    }

    def _convert(self, obj, store=None):
        from sktime.datatypes._adapter.dask_to_pd import convert_pandas_to_dask

        obj = _coerce_df_dtypes(obj)
        return convert_pandas_to_dask(obj)


class HierarchicalPolarsToPdMultiIndex(BaseConverter):
    """Conversion: polars_hierarchical -> pd_multiindex_hier, Hierarchical scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "polars_hierarchical",
        "mtype_to": "pd_multiindex_hier",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": "polars",
    }

    def _convert(self, obj, store=None):
        from sktime.datatypes._adapter.polars import convert_polars_to_pandas

        return convert_polars_to_pandas(obj)


class HierarchicalPdMultiIndexToPolars(BaseConverter):
    """Conversion: pd_multiindex_hier -> polars_hierarchical, Hierarchical scitype."""

    _tags = {
        "object_type": "converter",
        "mtype_from": "pd_multiindex_hier",
        "mtype_to": "polars_hierarchical",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": "polars",
    }

    def _convert(self, obj, store=None):
        from sktime.datatypes._adapter.polars import convert_pandas_to_polars

        return convert_pandas_to_polars(obj)


class HierarchicalDaskToPolars(BaseConverter):
    """Conversion: dask_hierarchical -> polars_hierarchical, of Hierarchical scitype.

    Converts via pd_multiindex_hier.
    """

    _tags = {
        "object_type": "converter",
        "mtype_from": "dask_hierarchical",
        "mtype_to": "polars_hierarchical",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": ["dask", "polars"],
    }

    def _convert(self, obj, store=None):
        obj = HierarchicalDaskToPdMultiIndex()(obj, store=store)
        return HierarchicalPdMultiIndexToPolars()(obj, store=store)


class HierarchicalPolarsToDask(BaseConverter):
    """Conversion: polars_hierarchical -> dask_hierarchical, of Hierarchical scitype.

    Converts via pd_multiindex_hier.
    """

    _tags = {
        "object_type": "converter",
        "mtype_from": "polars_hierarchical",
        "mtype_to": "dask_hierarchical",
        "multiple_conversions": False,
        "python_version": None,
        "python_dependencies": ["dask", "polars"],
    }

    def _convert(self, obj, store=None):
        obj = HierarchicalPolarsToPdMultiIndex()(obj, store=store)
        return HierarchicalPdMultiIndexToDask()(obj, store=store)
