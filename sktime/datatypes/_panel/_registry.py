# -*- coding: utf-8 -*-

__all__ = ["MTYPE_REGISTER_PANEL", "MTYPE_LIST_PANEL", "PanelMtype"]

import pandas as pd

from sktime.base._registry_enum import BaseRegistryEnum


class PanelMtype(BaseRegistryEnum):
    """Enum class for Panel mtypes."""

    NESTED_UNIV = (
        "nested_univ",
        "pd.DataFrame with one column per variable, pd.Series in cells",
        "Panel",
    )
    NUMPY3D = (
        "numpy3D",
        "3D np.array of format (n_instances, n_columns, n_timepoints)",
        "Panel",
    )
    NUMPYFLAT = (
        "numpyflat",
        "2D np.array of format (n_instances, n_columns*n_timepoints)",
        "Panel",
    )
    PD_MULTIINDEX = (
        "pd-multiindex",
        "pd.DataFrame with multi-index (instances, timepoints)",
        "Panel",
    )
    PD_WIDE = (
        "pd-wide",
        "pd.DataFrame in wide format, cols = (instance*timepoints)",
        "Panel",
    )
    PD_LONG = (
        "pd-long",
        "pd.DataFrame in long format, cols = (index, time_index, column)",
        "Panel",
    )
    DF_LIST = ("df-list", "list of pd.DataFrame", "Panel")


MTYPE_REGISTER_PANEL = [tuple(mtype) for mtype in PanelMtype]

MTYPE_LIST_PANEL = pd.DataFrame(MTYPE_REGISTER_PANEL)[0].values
