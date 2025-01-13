import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.hierarchical.aggregate import Aggregator

# TODO(felipeangelimvieira): Add summing matrix from series function
# TODO(felipeangelimvieira): Compute non-constrained reconciliation
# TODO(felipeangelimvieira): Compute non-negative reconciliation
# TODO(felipeangelimvieira): Immutable series reconciliation: should this be a separate class?

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


class FullHierarchyReconciler(BaseTransformer):

    _tags = _COMMON_TAGS

    def __init__(self, error_covariance_matrix: pd.DataFrame = None):
        self.error_covariance_matrix = error_covariance_matrix
        super().__init__()

    def _fit(self, X, y):
        self.aggregator_ = Aggregator()
        self.aggregator_.fit(X)
        self.unique_series_ = X.index.droplevel(-1).unique()
        self.S_ = create_summing_matrix_from_index(self.unique_series_)

        self._sigma = self.error_covariance_matrix.copy()
        if self.error_covariance_matrix is None:
            self._sigma = np.eye(len(self.unique_series_))
            self._sigma = pd.DataFrame(
                self._sigma,
                index=self.S_.index,
                columns=self.S_.index,
            )

    def _transform(self, X, y):

        X = self.aggregator_.transform(X)
        return X

    def _inverse_transform(self, X, y=None): ...


class NonNegativeHierarchyReconciler(BaseTransformer):
    _tags = _COMMON_TAGS

    def __init__(self, error_covariance_matrix: pd.DataFrame = None):
        self.error_covariance_matrix = error_covariance_matrix
        super().__init__()


import pandas as pd
import numpy as np


def create_summing_matrix_from_index(hier_index):
    """
    Given a MultiIndex 'hier_index' of a hierarchical time series
    (following an sktime-like convention), return a summation matrix S
    as a DataFrame. Each row corresponds to a node in the hierarchy,
    and each column corresponds to a bottom (leaf) node.
    The entry S[i, j] = 1 if row i (an aggregator node) is an ancestor
    of column j (a bottom node), else 0.
    """

    # Convert index to list of tuples for convenience
    all_nodes = list(hier_index)

    # Identify bottom nodes (leaf series): those with no '__total' in their tuple
    bottom_nodes = [node for node in all_nodes if "__total" not in node]

    # Initialize an (N x M) matrix: N = total nodes, M = bottom nodes
    N = len(all_nodes)
    M = len(bottom_nodes)
    S = np.zeros((N, M), dtype=int)

    # Helper function: check if 'agg' is an ancestor of 'bot'
    # Ancestor means that for each level `a_level` in agg and `b_level` in bot:
    #    a_level == '__total' OR a_level == b_level
    def is_ancestor(agg, bot):
        return all(a == b or a == "__total" for a, b in zip(agg, bot))

    # Populate the summation matrix
    for i, agg_node in enumerate(all_nodes):
        for j, bottom_node in enumerate(bottom_nodes):
            if is_ancestor(agg_node, bottom_node):
                S[i, j] = 1

    # Create a DataFrame with the same row index and MultiIndex columns
    S_df = pd.DataFrame(
        S,
        index=pd.MultiIndex.from_tuples(all_nodes, names=hier_index.names),
        columns=pd.MultiIndex.from_tuples(bottom_nodes, names=hier_index.names),
    )

    return S_df
