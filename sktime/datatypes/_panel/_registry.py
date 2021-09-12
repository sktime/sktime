# -*- coding: utf-8 -*-
__all__ = [
    "MTYPE_REGISTER_PANEL",
    "MTYPE_LIST_PANEL",
    "PanelMtype"
]

import pandas as pd
from sktime.registry._registry_enum import BaseRegistryEnum


class PanelMtype(BaseRegistryEnum):
    DF_LIST = (
        'df-list',
        'Panel',
        'list of pd.Dataframes'
    )
    NESTED_UNIV = (
        'nested_univ',
        'Panel',
        'pd.DataFrame with one column per variable, pd.Series in cells'
    )
    NUMPY_3D = (
        'numpy3D',
        'Panel',
        '3D np.array of format (n_instances, n_columns, n_timepoints)',
    )
    NUMPY_FLAT = (
        'numpyflat',
        'Panel',
        '2D np.array of format (n_instances, n_columns*n_timepoints)',
    )
    PD_LONG = (
        'pd-long',
        'Panel',
        'pd.DataFrame in long format, cols = (index, time_index, column)',
    )
    PD_MULTIINDEX = (
        'pd-multiindex',
        'Panel',
        'pd.DataFrame with multi-index (instances, timepoints)'
    )
    PD_WIDE = (
        'pd-wide',
        'Panel',
        'pd.DataFrame in wide format, cols = (instance*timepoints)'
    )


MTYPE_REGISTER_PANEL = [tuple(mtype) for mtype in PanelMtype]

MTYPE_LIST_PANEL = pd.DataFrame(MTYPE_REGISTER_PANEL)[0].values
