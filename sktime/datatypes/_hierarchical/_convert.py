# -*- coding: utf-8 -*-

__all__ = [
    "convert_dict",
]

from sktime.datatypes._convert_utils._convert import _extend_conversions
from sktime.datatypes._hierarchical._registry import MTYPE_LIST_HIERARCHICAL
from sktime.utils.validation._dependencies import _check_soft_dependencies

# dictionary indexed by triples of types
#  1st element = convert from - type
#  2nd element = convert to - type
#  3rd element = considered as this scitype - string
# elements are conversion functions of machine type (1st) -> 2nd

convert_dict = dict()


def convert_identity(obj, store=None):

    return obj


# assign identity function to type conversion to self
for tp in MTYPE_LIST_HIERARCHICAL:
    convert_dict[(tp, tp, "Hierarchical")] = convert_identity


if _check_soft_dependencies("dask", severity="none"):
    from sktime.datatypes._adapter.dask_to_pd import (
        convert_dask_to_pandas,
        convert_pandas_to_dask,
    )

    def convert_dask_to_pd_as_hierarchical(obj, store=None):
        return convert_dask_to_pandas(obj)

    convert_dict[
        ("dask_hierarchical", "pd_multiindex_hier", "Hierarchical")
    ] = convert_dask_to_pd_as_hierarchical

    def convert_pd_to_dask_as_hierarchical(obj, store=None):
        return convert_pandas_to_dask(obj)

    convert_dict[
        ("pd_multiindex_hier", "dask_hierarchical", "Hierarchical")
    ] = convert_pd_to_dask_as_hierarchical

    _extend_conversions(
        "dask_hierarchical",
        "pd_multiindex_hier",
        convert_dict,
        mtype_universe=MTYPE_LIST_HIERARCHICAL,
    )
