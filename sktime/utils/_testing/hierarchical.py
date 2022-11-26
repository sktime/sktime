#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Hierarchical Data Generators."""

__author__ = ["ltsaprounis", "ciaran-g"]

from itertools import product
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.datasets import load_airline
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


def _bottom_hier_datagen(
    no_levels=3,
    no_bottom_nodes=6,
    intercept_max=20,
    coef_1_max=20,
    coef_2_max=0.1,
    random_seed=None,
):
    """Hierarchical data generator using the flights dataset.

    This function generates bottom level, i.e. not aggregated, time-series
    from the flights dataset.

    Each series is generated from the flights dataset using a linear model,
    y = c0 + c1x + c2x^(c3), where the coefficients, intercept, and exponent
    are randomly sampled for each series. The coefficients and intercept are
    sampled between np.arange(0, *_max, 0.01) to keep the values positive. The
    exponent is sampled from [0.5, 1, 1.5, 2].


    Parameters
    ----------
    no_levels : int, optional
        The number of levels not considering the time-index, by default 3
    no_bottom_nodes : int, optional
       Number of time series, i.e. bottom nodes, to generate, by default 6.
    *_max : int, optional
        Maximum possible value of the coefficient or intercept value.
    random_seed : int, optional
        Random seed for reproducability.


    Returns
    -------
    pd.DataFrame with multiindex
    """
    if no_levels > no_bottom_nodes:
        raise ValueError("no_levels should be less than no_bottom_nodes")

    rng = np.random.default_rng(random_seed)

    base_ts = load_airline()
    df = pd.DataFrame(base_ts, index=base_ts.index)
    df.index.rename(None, inplace=True)

    if no_levels == 0:
        df.columns = ["passengers"]
        df.index.rename("timepoints", inplace=True)
        return df
    else:

        df.columns = ["l1_node01"]

        intercept = np.arange(0, intercept_max, 0.01)
        coef_1 = np.arange(0, coef_1_max, 0.01)
        coef_2 = np.arange(0, coef_2_max, 0.01)
        power_2 = [0.5, 1, 1.5, 2]

        # create structure of hierarchy
        node_lookup = pd.DataFrame(
            ["l1_node" + f"{x:02d}" for x in range(1, no_bottom_nodes + 1)]
        )
        node_lookup.columns = ["l1_agg"]

        if no_levels >= 2:

            # create index from bottom up, sampling node names
            for i in range(2, no_levels + 1):
                node_lookup["l" + str(i) + "_agg"] = node_lookup.groupby(
                    ["l" + str(i - 1) + "_agg"]
                )["l1_agg"].transform(
                    lambda x: "l"
                    + str(i)
                    + "_node"
                    + "{:02d}".format(_sample_node(node_lookup.index, i, rng))
                )

        node_lookup = node_lookup.set_index("l1_agg", drop=True)

        # now define the series for each level by sampling coefficients etc.
        for i in range(2, no_bottom_nodes + 1):
            df["l1_node" + f"{i:02d}"] = (
                rng.choice(intercept, size=1)
                + rng.choice(coef_1, size=1) * df["l1_node01"]
                + (
                    rng.choice(coef_2, size=1)
                    * (df["l1_node01"] ** rng.choice(power_2, size=1))
                )
            )

        df = (
            df.melt(ignore_index=False)
            .reset_index(drop=False)
            .rename(
                columns={
                    "variable": "l1_agg",
                    "index": "timepoints",
                    "value": "passengers",
                }
            )
        )

        df = pd.merge(left=df, right=node_lookup.reset_index(), on="l1_agg")
        df = df[df.columns.sort_values(ascending=True)]

        df_newindex = ["l" + str(x) + "_agg" for x in range(1, no_levels + 1)][::-1]
        df_newindex.append("timepoints")

        df = df.set_index(df_newindex)
        df.sort_index(inplace=True)

        return df


def _sample_node(index_table, level, sampler):
    """Sample a number of nodes depending on the size of hierarchy and level."""
    nodes = np.arange(1, np.floor(len(index_table) / level) + 1, 1)
    # return a single sample of them
    sample_nodes = int(sampler.choice(nodes, size=1))

    return sample_nodes
