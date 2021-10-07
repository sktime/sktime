# -*- coding: utf-8 -*-

__all__ = ["MTYPE_REGISTER_PANEL", "MTYPE_LIST_PANEL", "PanelMtype"]

import numpy as np
import pandas as pd

from sktime.base._registry_enum import BaseRegistryEnum


class PanelMtype(BaseRegistryEnum):
    """Enum class for Panel mtypes."""

    pd_univariate_nested_dataframe = (
        "pd.DataFrame with one column per variable, pd.Series in cells",
        "Panel",
        pd.DataFrame,
    )
    np_3d_array = (
        "3D np.array of format (n_instances, n_columns, n_timepoints)",
        "Panel",
        np.ndarray,
    )
    np_flat = (
        "2D np.array of format (n_instances, n_columns*n_timepoints)",
        "Panel",
        np.ndarray,
    )
    pd_multi_index_dataframe = (
        "pd.DataFrame with multi-index (instances, timepoints)",
        "Panel",
        pd.DataFrame,
    )
    pd_wide_dataframe = (
        "pd.DataFrame in wide format, cols = (instance*timepoints)",
        "Panel",
        pd.DataFrame,
    )
    pd_long_dataframe = (
        "pd.DataFrame in long format, cols = (index, time_index, column)",
        "Panel",
        pd.DataFrame,
    )
    list_pd_dataframe = ("list of pd.DataFrame", "Panel", list[pd.DataFrame])


MTYPE_REGISTER_PANEL = [tuple(mtype) for mtype in PanelMtype]

MTYPE_LIST_PANEL = pd.DataFrame(MTYPE_REGISTER_PANEL)[0].values
