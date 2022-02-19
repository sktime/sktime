#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hierarchical Data Generators."""

__author__ = ["ltsaprounis"]

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.utils._testing.series import _make_index


def _make_hierachical(
    hierarchy_levels: Tuple = (2, 4),
    n_timepoints: int = 12,
    n_columns: int = 1,
    all_positive: bool = True,
    index_type: str = None,
    random_state: int = None,
    add_nan: bool = False,
) -> pd.DataFrame:
    """Generate hierarchical multiindex mtype for testing."""
    levels = [
        [f"h{i}_{j}" for j in range(hierarchy_levels[i])]
        for i in range(len(hierarchy_levels))
    ]
    level_names = [f"h{i}" for i in range(len(hierarchy_levels))]
    time_index = _make_index(n_timepoints, index_type)
    index = pd.MultiIndex.from_product(
        levels + [time_index], names=level_names + ["time"]
    )
    total_time_points = len(index)
    rng = check_random_state(random_state)
    data = rng.normal(size=(total_time_points, n_columns))
    if add_nan:
        # add some nan values
        data[int(len(data) / 2)] = np.nan
        data[0] = np.nan
        data[-1] = np.nan
    if all_positive:
        data -= np.min(data, axis=0) - 1
    df = pd.DataFrame(
        data=data, index=index, columns=[f"c{i}" for i in range(n_columns)]
    )

    return df
