#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hierarchical Data Generators."""

__author__ = ["ltsaprounis"]

from itertools import product
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.utils._testing.series import _make_index


def _make_hierarchical(
    hierarchy_levels: Tuple = (2, 4),
    max_timepoints: int = 12,
    min_timepoints: int = 12,
    same_cutoff: bool = True,
    n_columns: int = 1,
    all_positive: bool = True,
    index_type: str = None,
    random_state: Union[int, np.random.RandomState] = None,
    add_nan: bool = False,
) -> pd.DataFrame:
    """Generate hierarchical multiindex mtype for testing.

    Parameters
    ----------
    hierarchy_levels : Tuple, optional
        the number of groups at each hierarchy level, by default (2, 4)
    max_timepoints : int, optional
        maximum time points a series can have, by default 12
    min_timepoints : int, optional
        minimum time points a seires can have, by default 12
    same_cutoff : bool, optional
        If it's True all series will end at the same date, by default True
    n_columns : int, optional
        number of columns in the output dataframe, by default 1
    all_positive : bool, optional
        If True the time series will be , by default True
    index_type : str, optional
        type of index, by default None
        Supported types are "period", "datetime", "range" or "int".
        If it's not provided, "datetime" is selected.
    random_state : int, np.random.RandomState or None
        Controls the randomness of the estimator, by default None
    add_nan : bool, optional
        If it's true the series will contain NaNs, by default False

    Returns
    -------
    pd.DataFrame
        hierarchical mtype dataframe
    """
    levels = [
        [f"h{i}_{j}" for j in range(hierarchy_levels[i])]
        for i in range(len(hierarchy_levels))
    ]
    level_names = [f"h{i}" for i in range(len(hierarchy_levels))]
    rng = check_random_state(random_state)
    if min_timepoints == max_timepoints:
        time_index = _make_index(max_timepoints, index_type)
        index = pd.MultiIndex.from_product(
            levels + [time_index], names=level_names + ["time"]
        )
    else:
        df_list = []
        for levels_tuple in product(*levels):
            n_timepoints = rng.randint(low=min_timepoints, high=max_timepoints)
            if same_cutoff:
                time_index = _make_index(max_timepoints, index_type)[-n_timepoints:]
            else:
                time_index = _make_index(n_timepoints, index_type)
            d = dict(zip(level_names, levels_tuple))
            d["time"] = time_index
            df_list.append(pd.DataFrame(d))
        index = pd.MultiIndex.from_frame(
            pd.concat(df_list), names=level_names + ["time"]
        )

    total_time_points = len(index)
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
