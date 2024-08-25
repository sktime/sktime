"""Registry of mtypes for Panel scitype.

See datatypes._registry for API.
"""

__all__ = [
    "MTYPE_REGISTER_PANEL",
    "MTYPE_LIST_PANEL",
    "MTYPE_SOFT_DEPS_PANEL",
]


MTYPE_REGISTER_PANEL = [
    (
        "nested_univ",
        "Panel",
        "pd.DataFrame with one column per variable, pd.Series in cells",
    ),
    (
        "numpy3D",
        "Panel",
        "3D np.array of format (n_instances, n_columns, n_timepoints)",
    ),
    (
        "numpyflat",
        "Panel",
        "WARNING: only for internal use, not a fully supported Panel mtype. "
        "2D np.array of format (n_instances, n_columns*n_timepoints)",
    ),
    ("pd-multiindex", "Panel", "pd.DataFrame with multi-index (instances, timepoints)"),
    ("pd-wide", "Panel", "pd.DataFrame in wide format, cols = (instance*timepoints)"),
    (
        "pd-long",
        "Panel",
        "pd.DataFrame in long format, cols = (index, time_index, column)",
    ),
    ("df-list", "Panel", "list of pd.DataFrame"),
    (
        "gluonts_ListDataset_panel",
        "Panel",
        "gluonTS representation of univariate and multivariate time series",
    ),
    (
        "gluonts_PandasDataset_panel",
        "Panel",
        "gluonTS representation of a pandas DataFrame",
    ),
]

MTYPE_SOFT_DEPS_PANEL = {
    "xr.DataArray": "xarray",
    "dask_panel": "dask",
    "gluonts_ListDataset_panel": "gluonts",
    "gluonts_PandasDataset_panel": "gluonts",
}

MTYPE_LIST_PANEL = [x[0] for x in MTYPE_REGISTER_PANEL]
