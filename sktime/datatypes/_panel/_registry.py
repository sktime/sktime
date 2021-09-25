# -*- coding: utf-8 -*-

__all__ = ["MTYPE_REGISTER_PANEL", "MTYPE_LIST_PANEL", "PanelMtype"]

import pandas as pd

from sktime.datatypes._mtype_enum import MtypeEnum


class PanelMtype(MtypeEnum):
    """Enum class for Panel mtypes."""

    nested_univariate = (
        "nested_univ",
        "pd.DataFrame with one column per variable, pd.Series in cells",
        "Panel",
        False,
    )
    np_3d_array = (
        "numpy3D",
        "3D np.array of format (n_instances, n_columns, n_timepoints)",
        "Panel",
        True,
    )
    np_flat = (
        "numpyflat",
        "2D np.array of format (n_instances, n_columns*n_timepoints)",
        "Panel",
        True,
    )
    pd_multi_index = (
        "pd-multiindex",
        "pd.DataFrame with multi-index (instances, timepoints)",
        "Panel",
        False,
    )
    pd_wide_df = (
        "pd-wide",
        "pd.DataFrame in wide format, cols = (instance*timepoints)",
        "Panel",
        False,
    )
    pd_long_df = (
        "pd-long",
        "pd.DataFrame in long format, cols = (index, time_index, column)",
        "Panel",
        False,
    )
    list_df = ("df-list", "list of pd.DataFrame", "Panel", False)


MTYPE_REGISTER_PANEL = [tuple(mtype) for mtype in PanelMtype]

MTYPE_LIST_PANEL = pd.DataFrame(MTYPE_REGISTER_PANEL)[0].values
