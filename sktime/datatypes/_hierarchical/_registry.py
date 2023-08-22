"""Registry of mtypes for Hierarchical scitype.

See datatypes._registry for API.
"""

__all__ = [
    "MTYPE_REGISTER_HIERARCHICAL",
    "MTYPE_LIST_HIERARCHICAL",
    "MTYPE_SOFT_DEPS_HIERARCHICAL",
]


MTYPE_REGISTER_HIERARCHICAL = [
    (
        "pd_multiindex_hier",
        "Hierarchical",
        "pd.DataFrame with MultiIndex",
    ),
    (
        "dask_hierarchical",
        "Hierarchical",
        "dask frame with multiple hierarchical indices, as per dask_to_pd convention",
    ),
]

MTYPE_SOFT_DEPS_HIERARCHICAL = {"dask_hierarchical": "dask"}

MTYPE_LIST_HIERARCHICAL = [x[0] for x in MTYPE_REGISTER_HIERARCHICAL]
