# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a transfromer to generate hierarcical data from bottom level."""

__author__ = ["ciaran-g"]

from warnings import warn

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
        "X_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,  # does transformer have inverse
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def __init__(self, flatten_single_levels=True):

        self.flatten_single_levels = flatten_single_levels

        super(Aggregator, self).__init__()

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
        X = X.copy()
        X_orig_idx_names = X.index.names
        new_idx_names = [str(x) for x in range(len(X_orig_idx_names))]
        X.index.names = new_idx_names

        if X.index.nlevels == 1:
            warn(
                "Aggregator is intended for use with X.index.nlevels > 1. "
                "Returning X unchanged."
            )
            return X
        else:
            # check the tests are ok
            if not _check_index_good(X):
                raise ValueError(
                    """Please check the index of X
                        1) Does not contain any elements named "__total".
                        2) Has all named levels.
                    """
                )
            else:
                pass

            # names from index
            hier_names = list(X.index.names)

            # top level
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

                    mid = X.groupby(level=agg_levels).sum()
                    # now fill in index
                    ind_names = list(set(hier_names).difference(agg_levels))
                    for j in ind_names:
                        mid[j] = "__total"
                    # set back in index
                    mid = mid.set_index(ind_names, append=True).reorder_levels(
                        hier_names
                    )
                    df_out = pd.concat([df_out, mid])

            # now remove duplicated aggregate indexes
            if self.flatten_single_levels:
                new_index = _flatten_single_indexes(X)
                nm = X.index.names[-1]
                # uncomment if contraints are relaxed on naming the time index
                # if nm is None:
                #     nm = "index"
                df_out = (
                    df_out.reset_index(level=-1)
                    .loc[new_index]
                    .set_index(nm, append=True, verify_integrity=True)
                )

            for i in range(df_out.index.nlevels - 1):

                idx = df_out.index.get_level_values(level=i)
                df_out = (
                    df_out.droplevel(i)
                    .set_index(idx, append=True)
                    .reorder_levels(hier_names)
                )

            df_out.sort_index(inplace=True)

            df_out.index.names = X_orig_idx_names
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

    # check the elements of the index for "__total"
    chk_list = []
    for i in list(X.index.names)[:-1]:
        chk_list.append(X.index.get_level_values(level=i).isin(["__total"]).sum())
    tot_chk = sum(chk_list) == 0

    all_ok = np.logical_and.reduce([nm_chk, tot_chk])

    return all_ok


def _flatten_single_indexes(X):
    """Check the index of X and return new unique index object."""
    # get unique indexes outwith timepoints
    inds = list(X.droplevel(-1).index.unique())
    ind_df = pd.DataFrame(inds, columns=X.droplevel(-1).index.unique().names)

    # add the new top aggregate level
    if len(ind_df.columns) == 1:
        out_list = ["__total"]
    else:
        out_list = [tuple(np.repeat("__total", len(ind_df.columns)))]

        # for each level check there are child nodes of length >1
        for i in range(1, len(ind_df.columns)):
            # all levels from top
            ind_aggs = ind_df.loc[:, ind_df.columns[0:-i:]]
            # filter and check for child nodes with only 1 nunique name
            if len(ind_aggs.columns) > 1:
                filter_cols = list(ind_aggs.columns[0:-1])
                filter_inds = ind_aggs.groupby(
                    by=filter_cols, as_index=False
                ).transform(lambda x: x.nunique())
                filter_inds = filter_inds[(filter_inds > 1)].dropna().index
                ind_aggs = ind_aggs.iloc[filter_inds, :]
            else:
                pass

            tmp = ind_aggs.groupby(by=list(ind_aggs.columns)).size()

            # get idex of these nodes
            agg_ids = list(tmp[tmp > 1].dropna().index)

            # add the aggregate label down the the length of the orginal index
            # only id add if there are two elements in list
            if len(agg_ids) > 1:

                agg_ids = [tuple([x]) if type(x) is not tuple else x for x in agg_ids]
                for _j in range(i):
                    agg_ids = [x + ("__total",) for x in agg_ids]

                out_list.extend(agg_ids)
            else:
                pass

    # add to original index
    inds.extend(out_list)

    if len(ind_df.columns) == 1:
        new_index = pd.Index(inds, name=ind_df.columns[0])
    else:
        new_index = pd.MultiIndex.from_tuples(
            inds,
            names=ind_df.columns,
        )

    return new_index
