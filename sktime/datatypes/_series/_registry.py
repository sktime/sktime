"""Registry of mtypes for Series scitype.

See datatypes._registry for API.
"""

__all__ = [
    "MTYPE_REGISTER_SERIES",
    "MTYPE_LIST_SERIES",
    "MTYPE_SOFT_DEPS_SERIES",
]


MTYPE_REGISTER_SERIES = [
    ("pd.Series", "Series", "pd.Series representation of a univariate series"),
    (
        "pd.DataFrame",
        "Series",
        "pd.DataFrame representation of a uni- or multivariate series",
    ),
    (
        "np.ndarray",
        "Series",
        "2D numpy.ndarray with rows=samples, cols=variables, index=integers",
    ),
    (
        "xr.DataArray",
        "Series",
        "xr.DataArray representation of a uni- or multivariate series",
    ),
    (
        "dask_series",
        "Series",
        "xdas representation of a uni- or multivariate series",
    ),
]

MTYPE_SOFT_DEPS_SERIES = {"xr.DataArray": "xarray", "dask_series": "dask"}

MTYPE_LIST_SERIES = [x[0] for x in MTYPE_REGISTER_SERIES]
