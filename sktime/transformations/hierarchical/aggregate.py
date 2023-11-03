# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a transformer to generate hierarchical data from bottom level."""

__author__ = ["ciaran-g"]

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.warnings import warn

# todo: add any necessary sktime internal imports here


class Aggregator(BaseTransformer):
    """Prepare hierarchical data, including aggregate levels, from bottom level.

    This transformer adds aggregate levels via summation to a DataFrame with a
    multiindex. The aggregate levels are included with the special tag "__total"
    in the index. The aggregate nodes are discovered from top-to-bottom from
    the input data multiindex.

    Parameters
    ----------
    flatten_single_level : boolean (default=True)
        Remove aggregate nodes, i.e. ("__total"), where there is only a single
        child to the level

    See Also
    --------
    ReconcilerForecaster
    Reconciler

    References
    ----------
    .. [1] https://otexts.com/fpp3/hierarchical.html

    Examples
    --------
    >>> from sktime.transformations.hierarchical.aggregate import Aggregator
    >>> from sktime.utils._testing.hierarchical import _bottom_hier_datagen
    >>> agg = Aggregator()
    >>> y = _bottom_hier_datagen(
    ...     no_bottom_nodes=3,
    ...     no_levels=1,
    ...     random_seed=123,
    ... )
    >>> y = agg.fit_transform(y)
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

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Panel of pd.DataFrame data to be transformed.
        y : Ignored argument for interface compatibility.

        Returns
        -------
        Transformed version of X
        """
        if X.index.nlevels == 1:
            warn(
                "Aggregator is intended for use with X.index.nlevels > 1. "
                "Returning X unchanged.",
                obj=self,
            )
            return X
        # check the tests are ok
        if not _check_index_no_total(X):
            warn(
                "Found elements in the index of X named '__total'. Removing "
                "these levels and aggregating.",
                obj=self,
            )
            X = self._inverse_transform(X)

        # starting from top aggregate
        df_out = X.copy()
        for i in range(0, X.index.nlevels - 1, 1):
            # finding "__totals" parent/child from (up -> down)
            indx_grouper = np.arange(0, i, 1).tolist()
            indx_grouper.append(X.index.nlevels - 1)

            out = X.groupby(level=indx_grouper).sum()

            # get new index with aggregate levels to match with old
            new_idx = []
            for j in range(0, X.index.nlevels - 1, 1):
                if j in indx_grouper:
                    new_idx.append(out.index.get_level_values(j))
                else:
                    new_idx.append(["__total"] * len(out.index))

            # add in time index
            new_idx.append(out.index.get_level_values(-1))

            new_idx = pd.MultiIndex.from_arrays(new_idx, names=X.index.names)

            out = out.set_index(new_idx)

            df_out = pd.concat([out, df_out])

        # now remove duplicated aggregate indexes
        if self.flatten_single_levels:
            new_index = _flatten_single_indexes(X)

            nm = X.index.names[-1]
            if nm is None:
                nm = "level_" + str(X.index.nlevels - 1)
            else:
                pass

            # now reindex with new non-duplicated axis
            df_out = (
                df_out.reset_index(level=-1).loc[new_index].set_index(nm, append=True)
            ).rename_axis(X.index.names, axis=0)

        df_out = df_out.sort_index()

        return df_out

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : Panel of pd.DataFrame data to be inverse transformed.
        y : Ignored argument for interface compatibility.

        Returns
        -------
        Inverse transformed version of X.
        """
        if X.index.nlevels == 1:
            warn(
                "Aggregator is intended for use with X.index.nlevels > 1. "
                "Returning X unchanged.",
                obj=self,
            )
            return X
        if _check_index_no_total(X):
            warn(
                "Inverse is intended to be used with aggregated data. "
                "Returning X unchanged.",
                obj=self,
            )
        else:
            for i in range(X.index.nlevels - 1):
                X = X.drop(index="__total", level=i)
        return X

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


def _check_index_no_total(X):
    """Check the index of X and return boolean."""
    # check the elements of the index for "__total"
    chk_list = []
    for i in range(0, X.index.nlevels - 1, 1):
        chk_list.append(X.index.get_level_values(level=i).isin(["__total"]).sum())
    tot_chk = sum(chk_list) == 0

    return tot_chk


def _flatten_single_indexes(X):
    """Check the index of X and return new unique index object."""
    # get unique indexes outwith timepoints
    inds = list(X.droplevel(-1).index.unique())
    ind_df = pd.DataFrame(inds)

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

            # add the aggregate label down the the length of the original index

            # only add if >=1 elements in list and not at the 2nd aggregate level
            add_indicator1 = (i < (len(ind_df.columns) - 1)) & (len(agg_ids) >= 1)
            # or at the second most aggregate level and there are two aggs to add
            # or at the second most aggregate level and there is 1 agg to add,
            #   but the top level has more than one unique index
            add_indicator2 = (len(agg_ids) > 1) | (
                (len(agg_ids) == 1) & (ind_df.iloc[:, 0].nunique() > 1)
            )

            if add_indicator1 | add_indicator2:
                agg_ids = [tuple([x]) if type(x) is not tuple else x for x in agg_ids]
                for _j in range(i):
                    agg_ids = [x + ("__total",) for x in agg_ids]

                out_list.extend(agg_ids)
            else:
                pass

    # add to original index
    inds.extend(out_list)

    if len(ind_df.columns) == 1:
        new_index = pd.Index(inds, name=X.index.droplevel(-1).name)
    else:
        new_index = pd.MultiIndex.from_tuples(
            inds,
            names=X.index.droplevel(-1).names,
        )

    return new_index
