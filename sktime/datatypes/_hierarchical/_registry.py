# -*- coding: utf-8 -*-
"""Registry of mtypes for Hierarchical scitype. See datatypes._registry for API."""

import pandas as pd

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
]

MTYPE_SOFT_DEPS_HIERARCHICAL = {}

MTYPE_LIST_HIERARCHICAL = pd.DataFrame(MTYPE_REGISTER_HIERARCHICAL)[0].values
