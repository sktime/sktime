"""Single-level reconciliation."""

import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.reconciliation._utils import (
    _is_hierarchical_dataframe,
    filter_descendants,
    get_middle_level_aggregators,
)

__all__ = ["MiddleOutReconciler"]


class MiddleOutReconciler(BaseTransformer):
    """
    Reconciliation using a middle-out approach.

    Parameters
    ----------
    middle_level : int
        The level at which to split the hierarchy for reconciliation.
    middle_top_reconciler : BaseTransformer
        The transformer to use for the top part of the hierarchy (above the middle).
    middle_bottom_reconciler : BaseTransformer
        The transformer to use for each subtree below the middle-level totals.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
        # estimator type
        # --------------
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
        "capability:inverse_transform": True,  # does transformer have inverse
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        "capability:hierarchical_reconciliation": True,
    }

    def __init__(
        self,
        middle_level: int,
        middle_top_reconciler: BaseTransformer,
        middle_bottom_reconciler: BaseTransformer,
    ):
        self.middle_level = middle_level
        self.middle_top_reconciler = middle_top_reconciler
        self.middle_bottom_reconciler = middle_bottom_reconciler
        super().__init__()

    def _fit(self, X, y):
        self._no_hierarchy = not _is_hierarchical_dataframe(X)
        if self._no_hierarchy:
            return self

        # Identify sub-series for top approach, sub-series for each aggregator node
        # We'll just store the aggregator nodes here for later usage.
        self._middle_aggregators = get_middle_level_aggregators(X, self.middle_level)

        # For the top portion, we want everything that is "middle or above"
        # i.e. anything which has the next level as '__total'
        # or is above that. We'll simply pass them all to middle_top_reconciler.
        X_middle_top = X.loc[X.index.droplevel(-1).isin(self._middle_aggregators)]
        # Fit top reconciler on that portion
        self.middle_top_reconciler.fit(X_middle_top, y)

        # For the bottom approach, we don't just do one chunk — we do it
        # aggregator-by-aggregator in transform. But we can still .fit() them
        # in a combined manner if the bottom approach has no aggregator-level states
        # that conflict. If it does, you might prefer to do aggregator-by-aggregator.
        # We'll do a single pass that includes all nodes "middle or below."
        # A naive approach is to feed everything to the bottom reconciler.
        # Or you can skip if your bottom reconcilers only need .fit() on bottom series.
        self.middle_bottom_reconciler.fit(X, y)

        return self

    def _transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        # 1) Reconcile the middle-level aggregator nodes with the top approach
        X_middle_top = X.loc[X.index.droplevel(-1).isin(self._middle_aggregators)]
        X_middle_top_reconciled = self.middle_top_reconciler.transform(X_middle_top)

        # 2) For each middle-level aggregator node, get the subtree below it
        #    and apply the bottom approach
        bottom_subtrees = []
        for agg_node in self._middle_aggregators:
            X_subtree = filter_descendants(X, agg_node)
            if len(X_subtree) == 0:
                continue
            X_subtree_rec = self.middle_bottom_reconciler.transform(X_subtree)
            bottom_subtrees.append(X_subtree_rec)

        if bottom_subtrees:
            X_middle_bottom = pd.concat(bottom_subtrees, axis=0)
        else:
            X_middle_bottom = (
                pd.DataFrame([], columns=X.columns)
                if isinstance(X, pd.DataFrame)
                else pd.Series([], name=X.name, dtype=X.dtype)
            )

        # 3) Also, there may be nodes *above* the middle aggregator or
        #    not belonging to any aggregator. We typically include them in top part.
        #    So let's gather everything else that wasn't in aggregator subtrees
        #    or aggregator nodes themselves. One naive approach:
        #    We'll treat them with middle_top approach or just pass them unchanged.
        #    For a minimal fix, let's pass them to the top approach as well.
        #    We'll identify these 'remaining' nodes:
        used_idx = (
            X_middle_top_reconciled.index.droplevel(-1)
            .unique()
            .union(X_middle_bottom.index.droplevel(-1).unique())
        )
        # Everything not used yet:
        remaining_mask = ~X.index.droplevel(-1).isin(used_idx)
        X_remaining = X.loc[remaining_mask]

        # Reconcile or pass them as well
        X_remaining_rec = self.middle_top_reconciler.transform(X_remaining)

        # 4) Combine
        _X = pd.concat(
            [X_middle_top_reconciled, X_middle_bottom, X_remaining_rec], axis=0
        ).sort_index()

        # Remove duplicates if they exist
        _X = _X[~_X.index.duplicated(keep="first")]

        return _X

    def _inverse_transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        # The inverse transform logic is analogous:
        # 1) Middle-level aggregator nodes => top approach inverse
        X_middle_top = X.loc[X.index.droplevel(-1).isin(self._middle_aggregators)]
        X_middle_top_inv = self.middle_top_reconciler.inverse_transform(X_middle_top)

        # 2) For each aggregator node, get the subtree and do bottom approach inverse
        bottom_subtrees = []
        for agg_node in self._middle_aggregators:
            X_subtree = filter_descendants(X, agg_node)
            if len(X_subtree) == 0:
                continue
            X_subtree_inv = self.middle_bottom_reconciler.inverse_transform(X_subtree)
            bottom_subtrees.append(X_subtree_inv)

        if bottom_subtrees:
            X_middle_bottom_inv = pd.concat(bottom_subtrees, axis=0)
        else:
            X_middle_bottom_inv = (
                pd.DataFrame([], columns=X.columns)
                if isinstance(X, pd.DataFrame)
                else pd.Series([], name=X.name, dtype=X.dtype)
            )

        # 3) Handle anything not covered
        used_idx = (
            X_middle_top_inv.index.droplevel(-1)
            .unique()
            .union(X_middle_bottom_inv.index.droplevel(-1).unique())
        )
        remaining_mask = ~X.index.droplevel(-1).isin(used_idx)
        X_remaining = X.loc[remaining_mask]

        # Possibly pass that to top approach inverse or keep as-is:
        X_remaining_inv = self.middle_top_reconciler.inverse_transform(X_remaining)

        # 4) Combine
        _X = pd.concat(
            [X_middle_top_inv, X_middle_bottom_inv, X_remaining_inv], axis=0
        ).sort_index()

        _X = _X[~_X.index.duplicated(keep="first")]
        return _X

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get test params."""
        from sktime.transformations.hierarchical.reconciliation.bottom_up import (
            BottomUpReconciler,
        )
        from sktime.transformations.hierarchical.reconciliation.topdown_share import (
            TopdownShareReconciler,
        )

        return [
            {
                "middle_level": 1,
                "middle_top_reconciler": TopdownShareReconciler(),
                "middle_bottom_reconciler": BottomUpReconciler(),
            },
            {
                "middle_level": 2,
                "middle_top_reconciler": TopdownShareReconciler(),
                "middle_bottom_reconciler": BottomUpReconciler(),
            },
        ]
