"""Middle-out reconciler reconciliation."""

import warnings

import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.reconcile._base import _ReconcilerTransformer
from sktime.transformations.hierarchical.reconcile._bottom_up import (
    BottomUpReconciler,
)
from sktime.transformations.hierarchical.reconcile._topdown import (
    TopdownReconciler,
)
from sktime.transformations.hierarchical.reconcile._utils import (
    _filter_descendants,
    _get_series_for_each_hierarchical_level,
    _loc_series_idxs,
)

__all__ = ["MiddleOutReconciler"]


class MiddleOutReconciler(_ReconcilerTransformer):
    """
    Reconciliation using a middle-out approach.

    This reconciliation strategy splits the hierarchy at a given level and
    applies a bottom-up strategy to the top part of the hierarchy and a
    topdown strategy to the bottom part of the hierarchy.

    The parameter middle-level is determined by the level according to
    the hierarchy tree. For example, consider the following structure with four
    levels:

    ```
    __total
    ├── B1
    │   ├── C1
    │   │   ├── D1
    │   │   └── D2
    │   └── C2
    │       ├── D3
    │       └── D4
    └── B2
        ├── C3
        │   ├── D5
        │   └── D6
    ```

    If `middle_level` is set to 0, then the hierarchy is split at the root node.
    If `middle_level` is set to 1, then the hierarchy is split at the first level
    below the root node, in this example, the nodes [B1, B2].

    It is important to note that the height of this hierarchy tree don't
    necessarily coincides with the number of levels in the pd.DataFrame index.
    For example, the following index has 3 levels, but the tree has 4 levels:

    ```
    __total, __total, __total
    B1, __total, __total
    B1, C1, __total
    ...
    B2, __total, __total
    B2, C3, __total
    ...
    ```


    Parameters
    ----------
    middle_level : int
        The level at which to split the hierarchy for reconciliation.
    middle_bottom_reconciler : BaseTransformer
        The transformer to use for each subtree below the middle-level totals.

    Examples
    --------
    >>> from sktime.transformations.hierarchical.reconcile import (
    ...     MiddleOutReconciler)
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.hierarchical.aggregate import Aggregator
    >>> y = _make_hierarchical(hierarchy_levels=(2, 2, 4))
    >>> pipe = MiddleOutReconciler(middle_level=1) * NaiveForecaster()
    >>> pipe = pipe.fit(y)
    >>> y_pred = pipe.predict(fh=[1,2,3])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "felipeangelimvieira",
        "maintainers": "felipeangelimvieira",
    }

    def __init__(
        self,
        middle_level: int,
        middle_bottom_reconciler: BaseTransformer = None,
    ):
        self.middle_level = middle_level
        self.middle_bottom_reconciler = middle_bottom_reconciler
        super().__init__()

        self._middle_bottom_reconciler = self.middle_bottom_reconciler
        if self._middle_bottom_reconciler is None:
            self._middle_bottom_reconciler = TopdownReconciler()

        self._delegate = None

    def _fit_reconciler(self, X, y):
        """
        Fit the reconciler.

        This method does the following:

        - If this is the last level of the hierarchy, it delegates
        to bottom-up reconciliation
        - If this is the first level of the hierarchy, it delegates
        to top-down reconciliation
        - Otherwise, it splits the hierarchy at the middle level and detects,
        for each middle-level nodes, its descendants and to apply the
        reconciliation to them.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.
        y : pd.Series
            Exogenous data.

        Returns
        -------
        self : MiddleOutReconciler
            The fitted reconciler.
        """
        series_ids = X.index.droplevel(-1).unique()
        self._hierarchical_level_nodes = _get_series_for_each_hierarchical_level(
            series_ids
        )

        if (
            self.middle_level == len(self._hierarchical_level_nodes) - 1
            or self.middle_level == -1
        ):
            warnings.warn(
                "Middle level is the last level of the hierarchy."
                "Using bottom-up strategy only."
            )
            self._delegate = BottomUpReconciler()
            self._delegate.fit(X, y)
            return self

        if self.middle_level == 0 or self.middle_level == -len(
            self._hierarchical_level_nodes
        ):
            warnings.warn(
                "Middle level is the first level of the hierarchy. "
                "Using Topdown strategy only."
            )
            self._delegate = self.middle_bottom_reconciler.clone()
            self._delegate.fit(X, y)
            return self

        try:
            self.middle_level_series_ = self._hierarchical_level_nodes[
                self.middle_level
            ]
        except IndexError as e:
            n_levels = len(self._hierarchical_level_nodes)
            raise ValueError(
                f"middle_level should be between 0 and {n_levels - 1}, that is"
                f" the number of levels in the hierarchy."
                f" The provided value was {self.middle_level}."
            ) from e

        # Notice that the topdown strategy needs to be fitted for each
        # middle-level note, separately.
        self.middle_bottom_subtrees_ = self._get_middle_bottom_subtrees(X)
        self.middle_botttom_reconcilers_ = {}
        self.middle_bottom_drop_redundant_levels_ = {}
        for middle_node, descendants_idx in self.middle_bottom_subtrees_.items():
            X_subtree = _loc_series_idxs(X, descendants_idx)

            X_subtree = X_subtree.droplevel(
                level=list(range(middle_node.index("__total")))
            )

            y_subtree = y
            if y is not None:
                y_subtree = _loc_series_idxs(y, descendants_idx)
                y_subtree = y_subtree.droplevel(
                    level=list(range(middle_node.index("__total")))
                )
            _middle_bottom_reconciler = self._middle_bottom_reconciler.clone()

            _middle_bottom_reconciler.fit(X_subtree, y_subtree)
            self.middle_botttom_reconcilers_[middle_node] = _middle_bottom_reconciler

        return self

    def _get_middle_bottom_subtrees(self, X):
        """
        Get the subtrees below the middle-level nodes.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.

        Returns
        -------
        bottom_subtrees : dict
            A dictionary with the middle-level nodes as keys and the
            corresponding subtrees as values.
        """
        # 2) For each middle-level aggregator node, get the subtree below it
        #    and apply the bottom approach
        bottom_subtrees = {}
        for agg_node in self.middle_level_series_:
            X_subtree = _filter_descendants(X, agg_node)
            if len(X_subtree) == 0:
                continue
            bottom_subtrees[agg_node] = X_subtree.index.droplevel(-1).unique()
        return bottom_subtrees

    def _transform_reconciler(self, X, y=None):
        """
        Apply the Topdown from middle to bottom.

        Keeps the bottom levels according to the top-down strategy chosen.
        """
        if self._delegate is not None:
            return self._delegate.transform(X, y)

        X_middle = _loc_series_idxs(
            X, self._hierarchical_level_nodes[self.middle_level]
        )

        # 2) For each middle-level aggregator node, get the subtree below it
        #    and apply the bottom approach
        bottom_subtrees = []
        for agg_node in self.middle_level_series_:
            X_subtree = _filter_descendants(X, agg_node)
            if len(X_subtree) == 0:
                continue

            first_relevant_level = agg_node.index("__total")

            _idx = X_subtree.index

            X_subtree = X_subtree.droplevel(level=list(range(first_relevant_level)))

            y_subtree = y
            if y is not None:
                y_subtree = _loc_series_idxs(y, _idx.droplevel(-1))
                y_subtree = y_subtree.droplevel(level=list(range(first_relevant_level)))

            X_subtree_trans = self.middle_botttom_reconcilers_[agg_node].transform(
                X_subtree, y_subtree
            )

            X_subtree_trans.index = _idx.join(X_subtree_trans.index, how="right")
            X_subtree_trans = X_subtree_trans.reorder_levels(_idx.names)

            bottom_subtrees.append(X_subtree_trans)

        if bottom_subtrees:
            X_middle_bottom = pd.concat(bottom_subtrees, axis=0)
            Xt = X_middle_bottom.sort_index()
        else:
            Xt = X_middle.sort_index()

        return Xt

    def _inverse_transform_reconciler(self, X, y=None):
        """
        Apply top-down and obtain reconciled middle-bottom forecasts.

        The bottom-up is later applied in the parent reconciler class.
        """
        if self._delegate is not None:
            return self._delegate.inverse_transform(X, y)

        # 2) For each aggregator node, get the subtree and do bottom approach inverse
        bottom_subtrees = []
        for agg_node in self.middle_level_series_:
            X_subtree = _filter_descendants(X, agg_node)
            if len(X_subtree) == 0:
                continue

            first_relevant_level = agg_node.index("__total")

            _idx = X_subtree.index

            X_subtree = X_subtree.droplevel(level=list(range(first_relevant_level)))

            X_subtree_inv = self.middle_botttom_reconcilers_[
                agg_node
            ].inverse_transform(X_subtree)

            X_subtree_inv.index = _idx.join(X_subtree_inv.index, how="right")
            X_subtree_inv = X_subtree_inv.reorder_levels(_idx.names)
            bottom_subtrees.append(X_subtree_inv)

        _X = pd.concat(bottom_subtrees, axis=0)
        return _X

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Get test params."""
        from sktime.transformations.hierarchical.reconcile._topdown import (
            TopdownReconciler,
        )

        return [
            {
                "middle_level": 0,
                "middle_bottom_reconciler": TopdownReconciler(),
            },
            {
                "middle_level": -1,
                "middle_bottom_reconciler": TopdownReconciler(),
            },
        ]
