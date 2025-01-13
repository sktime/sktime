"""Single-level reconciliation."""

import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator

__all__ = ["TopdownShareReconciler", "BottomUpReconciler", "MiddleOutReconciler"]


def loc_series_idxs(y, idxs):
    return y.loc[y.index.droplevel(-1).isin(idxs)]


def get_bottom_level_idxs(y):
    idx = y.index
    idx = idx.droplevel(-1).unique()
    return idx[idx.get_level_values(-1) != "__total"]


def get_bottom_series(y):
    bottom_idx = get_bottom_level_idxs(y)
    return loc_series_idxs(y, bottom_idx)


def get_total_level_idxs(y):
    nlevels = y.index.droplevel(-1).nlevels
    return pd.Index([tuple(["__total"] * nlevels)])


def get_total_series(y):
    total_idx = get_total_level_idxs(y)
    return loc_series_idxs(y, total_idx)


def get_middle_level_series(y, middle_level):
    idx = y.index
    idx = idx.droplevel(-1).unique()
    idx_middle_level = idx.get_level_values(middle_level)
    idx_below_middle_level = idx.get_level_values(middle_level + 1)
    is_middle_level_mask = (idx_below_middle_level == "__total") & (
        idx_middle_level != "__total"
    )
    return idx[is_middle_level_mask]


def split_middle_levels(y, middle_level):
    """
    Return two series: middle level and above, and middle level and below.
    """
    idx = y.index
    idx = idx.droplevel(-1).unique()
    idx_middle_level = idx.get_level_values(middle_level)
    idx_below_middle_level = idx.get_level_values(middle_level + 1)
    is_middle_or_above_mask = idx_below_middle_level == "__total"
    is_middle_or_below_mask = idx_middle_level != "__total"

    return loc_series_idxs(y, idx[is_middle_or_above_mask]), loc_series_idxs(
        y, idx[is_middle_or_below_mask]
    )


_COMMON_TAGS = {
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
    "capability:inverse_transform": False,  # does transformer have inverse
    "skip-inverse-transform": True,  # is inverse-transform skipped when called?
    "univariate-only": False,  # can the transformer handle multivariate X?
    "handles-missing-data": False,  # can estimator handle missing data?
    "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
    "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
    "transform-returns-same-time-index": False,
}


class TopdownShareReconciler(BaseTransformer):
    """
    Topdown reconciliation.

    TopdownReconciler is a transformer for hierarchical time series reconciliation
    using the top-down approach.

    Attributes
    ----------
    _tags : dict
        Common tags for the transformer.
    _no_hierarchy : bool
        Indicates if the input series has no hierarchy.
    _original_series : pd.Index
        Original series index without the bottom level.
    _aggregator : Aggregator
        Aggregator instance to aggregate the series.
    _total_series : pd.Index
        Index of the total level series.
    _bottom_series : pd.Index
        Index of the bottom level series.
    """

    _tags = _COMMON_TAGS

    def _fit(self, X, y):
        self._no_hierarchy = X.index.nlevels == 1

        if self._no_hierarchy:
            return self

        self._original_series = X.index.droplevel(-1).unique()
        self._aggregator = Aggregator()
        self._aggregator.fit(X)
        X = self._aggregator.transform(X)

        self._total_series = get_total_level_idxs(X)
        self._bottom_series = get_bottom_level_idxs(X)

        return self

    def _transform(self, X, y):
        if self._no_hierarchy:
            return X

        X = self._aggregator.transform(X)
        X_total = loc_series_idxs(X, self._total_series)
        X_bottom = loc_series_idxs(X, self._bottom_series)

        X_total.index = X_total.index.get_level_values(-1)

        X_total_expanded = X_total.reindex(X_bottom.index.get_level_values(-1))
        X_total_expanded.index = X_bottom.index

        shares = X_bottom / X_total_expanded

        _X = pd.concat([loc_series_idxs(X, self._total_series), shares], axis=0)
        _X = loc_series_idxs(_X, self._original_series).sort_index()

        return _X

    def _inverse_transform(self, X, y):
        if self._no_hierarchy:
            return X

        X_total = loc_series_idxs(X, self._total_series)
        X_bottom = loc_series_idxs(X, self._bottom_series)

        # Keep only timeindex of total series
        X_total.index = X_total.index.get_level_values(-1)

        # Reindex total series to match bottom series
        X_total_expanded = X_total.reindex(X_bottom.index.get_level_values(-1))
        X_total_expanded.index = X_bottom.index

        forecasts_from_shares = X_bottom * X_total_expanded
        _X = pd.concat(
            [loc_series_idxs(X, self._original_series), forecasts_from_shares], axis=0
        )

        _X = self._aggregator.transform(_X)
        _X = loc_series_idxs(_X, self._original_series).sort_index()
        return _X


class BottomUpReconciler(BaseTransformer):
    """
    Bottom-up reconciliation for hierarchical time series data.

    It aggregates the data to the bottom level and then transforms it back
    to the original hierarchy.

    Attributes
    ----------
    _tags : dict
        Common tags for the transformer.
    _no_hierarchy : bool
        Indicates if the data has no hierarchy (single level).
    _original_series : pd.Index
        The original series index before transformation.
    _aggregator : Aggregator
        An instance of the Aggregator class used for aggregation.
    _bottom_series : pd.Index
        The index of the bottom level series.
    """

    _tags = _COMMON_TAGS

    def _fit(self, X, y):
        self._no_hierarchy = X.index.nlevels == 1

        if self._no_hierarchy:
            return self

        self._original_series = X.index.droplevel(-1).unique()
        self._aggregator = Aggregator()
        self._aggregator.fit(X)
        X = self._aggregator.transform(X)

        self._bottom_series = get_bottom_level_idxs(X)

        return self

    def _transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        X = self._aggregator.transform(X)
        X_bottom = loc_series_idxs(X, self._bottom_series)

        return X_bottom

    def _inverse_transform(self, X, y=None):
        if self._no_hierarchy:
            return X

        X = self._aggregator.transform(X)
        X = loc_series_idxs(X, self._original_series).sort_index()

        return X


from sktime.transformations.base import BaseTransformer

# other imports as needed


def get_middle_level_aggregators(X, middle_level):
    """
    Identify aggregator nodes at the specified middle_level, i.e.
    those for which the next level is '__total'.

    Example:
      If middle_level=1, then for a node= (region, store, cat, dept),
      we check node[middle_level+1] == '__total'.
      The node itself must not be __total at middle_level
      (otherwise it's above or not a well-defined middle node).
    """
    idx = X.index.droplevel(-1).unique()
    if middle_level + 1 >= idx.nlevels:
        # If the user picks a middle_level at the last level, there's no "next" level
        return []

    # The aggregator condition:
    #   * the next level's value is '__total'
    #   * the middle level's value != '__total'
    next_level_vals = idx.get_level_values(middle_level + 1)
    this_level_vals = idx.get_level_values(middle_level)

    is_agg = (next_level_vals == "__total") & (this_level_vals != "__total")
    return idx[is_agg]


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
        self._no_hierarchy = X.index.nlevels == 1
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


def is_ancestor(agg_node, node):
    """Returns True if agg_node is an ancestor of node."""
    return all(a == b or a == "__total" for a, b in zip(agg_node, node))


def filter_descendants(X, aggregator_node):
    """
    Returns a sub-DataFrame/Series of X containing only rows whose
    droplevel(-1) is a descendant of 'aggregator_node' at the given 'middle_level'.

    aggregator_node is a tuple like ('CA', 'CA_1', '__total', '__total')
    or ('regionA', 'storeA', '__total', '__total'), etc.
    """
    # We'll operate on the "higher-level" portion of the index
    # (i.e., the index after dropping the time or final level).
    idx_upper = X.index.droplevel(-1)  # e.g. (region, store, cat, dept)
    nodes = idx_upper.unique()

    # We only want to keep those which are descendants of aggregator_node,
    # i.e., aggregator_node is an ancestor of that node.
    # Because aggregator_node is "at" middle_level,
    # but it may also have __total at deeper levels.
    descendant_mask = [is_ancestor(aggregator_node, n) for n in nodes]
    descendant_nodes = nodes[descendant_mask]

    # Now filter X by these nodes
    return X.loc[idx_upper.isin(descendant_nodes)]
