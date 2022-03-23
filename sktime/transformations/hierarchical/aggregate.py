# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a transfromer to generate hierarcical data from bottom level."""

__author__ = ["ciaran-g"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

# todo: add any necessary sktime internal imports here


class aggregate_df(BaseTransformer):
    """Prepare hierarchical data, including aggregate levels, from bottom level.

    This transformer adds aggregate levels via summation to a DataFrame with a
    multiindex. The aggregate levels are included with the special tag "__total"
    in the index.

    Parameters
    ----------
    flatten_single_level : boolean (default=True)
        Remove aggregate levels, i.e. ("__total"), where there is only a single
        child in the level
    """

    _tags = {
        "scitype:transform-input": "Hierarchical",
        "scitype:transform-output": "Hierarchical",
        "scitype:transform-labels": "None",
        # todo instance wise?
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd_multiindex_hier",
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,  # does transformer have inverse
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": True,  # index type that needs to be enforced in X/y
        "fit-in-transform": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def __init__(self, flatten_single_levels=True):

        self.flatten_single_levels = flatten_single_levels

        super(aggregate_df, self).__init__()

    # todo: implement this, mandatory
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
            top = X.loc[mask1].groupby(level=-1).sum()

        ind_names = hier_names[:-1]
        for i in ind_names:
            top[i] = "__total"

        top = top.set_index(ind_names, append=True).reorder_levels(hier_names)

        df_out = pd.concat([top, X])

        # if we have a hierarchy with mid levels
        if len(hier_names) > 2:
            for i in range(len(hier_names) - 2):
                # list of levels to aggregate
                agg_levels = hier_names[0 : (i + 1)]
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

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        #
        # this can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # return params
