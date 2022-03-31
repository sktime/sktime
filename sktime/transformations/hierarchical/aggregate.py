# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a transfromer to generate hierarcical data from bottom level."""

__author__ = ["ciaran-g"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

# todo: add any necessary sktime internal imports here


class Aggregator(BaseTransformer):
    """Prepare hierarchical data, including aggregate levels, from bottom level.

    This transformer adds aggregate levels via summation to a DataFrame with a
    multiindex. The aggregate levels are included with the special tag "__total"
    in the index.

    Parameters
    ----------
    flatten_single_level : boolean (default=True)
        Remove aggregate nodes, i.e. ("__total"), where there is only a single
        child to the level
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        # todo instance wise?
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,  # does transformer have inverse
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit-in-transform": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def __init__(self, flatten_single_levels=True):

        self.flatten_single_levels = flatten_single_levels

        super(Aggregator, self).__init__()

    # todo: test that "__total" is not named in index?
    # todo: test that the index is named
    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Panel of mtype pd_multiindex_hier
            Data to be transformed
        y : Ignored argument for interface compatibility.

        Returns
        -------
        df_out : multi-indexed pd.DataFrame of Panel mtype pd_multiindex
        """
        # check the tests are ok
        if not _check_index_good(X):
            raise ValueError(
                """Please check the index of X
                    1) Does not contain any elements named "__total".
                    2) Has all named levels.
                    2) Has more than one level.
                    """
            )

        # names from index
        hier_names = list(X.index.names)

        # top level
        # remove aggregations that only have one level from below
        if self.flatten_single_levels:
            single_df = X.groupby(level=-1).count()
            mask1 = (
                single_df[(single_df > 1).all(1)]
                .index.get_level_values(level=-1)
                .unique()
            )
            mask1 = X.index.get_level_values(level=-1).isin(mask1)
            top = X.loc[mask1].groupby(level=-1).sum()
        else:
            top = X.groupby(level=-1).sum()

        ind_names = hier_names[:-1]
        for i in ind_names:
            top[i] = "__total"

        top = top.set_index(ind_names, append=True).reorder_levels(hier_names)

        df_out = pd.concat([top, X])

        # if we have a hierarchy with mid levels
        if len(hier_names) > 2:
            for i in range(len(hier_names) - 2):
                # list of levels to aggregate
                # aggregate from left index inward
                agg_levels = hier_names[0 : (i + 1)]
                # add in the final index (e.g. timepoints)
                agg_levels.append(hier_names[-1])

                # remove aggregations that only have one level from below
                if self.flatten_single_levels:
                    single_df = X.groupby(level=agg_levels).count()
                    # get index masks
                    masks = []
                    for i in agg_levels:
                        m1 = (
                            single_df[(single_df > 1).all(1)]
                            .index.get_level_values(i)
                            .unique()
                        )
                        m1 = X.index.get_level_values(i).isin(m1)
                        masks.append(m1)

                    mid = (
                        X.loc[np.logical_and.reduce(masks)]
                        .groupby(level=agg_levels)
                        .sum()
                    )
                else:
                    mid = X.groupby(level=agg_levels).sum()

                # now fill in index
                ind_names = list(set(hier_names).difference(agg_levels))
                for j in ind_names:
                    mid[j] = "__total"
                # set back in index
                mid = mid.set_index(ind_names, append=True).reorder_levels(hier_names)
                df_out = pd.concat([df_out, mid])

        df_out.sort_index(inplace=True)
        return df_out

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"flatten_single_levels": True}

        return params


def _check_index_good(X):
    """Check the index of X and return boolean."""
    # check the index is named
    ind_names = list(X.index.names)
    nm_chk = sum([y is not None for y in ind_names]) == len(ind_names)

    # check the length of index
    nmln_chk = len(ind_names) >= 2

    # check the elements of the index for "__total"
    chk_list = []
    for i in list(X.index.names)[:-1]:
        chk_list.append(X.index.get_level_values(level=i).isin(["__total"]).sum())
    tot_chk = sum(chk_list) == 0

    all_ok = np.logical_and.reduce([nm_chk, nmln_chk, tot_chk])

    return all_ok
