"""Full-hierarchy reconciliation."""

import numpy as np
import pandas as pd

from sktime.transformations.hierarchical.reconcile._base import _ReconcilerTransformer
from sktime.transformations.hierarchical.reconcile._utils import (
    _is_ancestor,
)

# TODO(felipeangelimvieira): Immutable series reconciliation: should this be
# a separate class?

__all__ = [
    "OptimalReconciler",
    "NonNegativeOptimalReconciler",
]


class OptimalReconciler(_ReconcilerTransformer):
    """
    Reconciliation for hierarchical time series.

    Uses all the forecasts to obtain a reconciled forecast.
    Uses the constraint matrix approach, which is more efficient than
    the projection one.

    If the dataframe is not hierarchical, this works as identity.

    Parameters
    ----------
    error_covariance_matrix : pd.DataFrame, default=None
        Error covariance matrix. If None, it is assumed to be the identity matrix.
    alpha : float, default=0
        Constant added to the diagonal of the inverted matrix.

    Examples
    --------
    >>> from sktime.transformations.hierarchical.reconcile import (
    ...     OptimalReconciler)
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = _make_hierarchical()
    >>> pipe = OptimalReconciler() * NaiveForecaster()
    >>> pipe = pipe.fit(y)
    >>> y_pred = pipe.predict(fh=[1,2,3])
    """

    def __init__(self, error_covariance_matrix: pd.DataFrame = None, alpha=0):
        self.error_covariance_matrix = error_covariance_matrix
        self.alpha = alpha
        super().__init__()

    def _fit_reconciler(self, X, y):
        """
        Fit optimal reconciler.

        Here, some important components are determined:

        - The summation matrix S, which maps bottom levels to all levels
        - The A and I sub-matrices of S, which map bottom levels to aggregated
        levels, and bottom levels to themselves, respectively.
        - The permutation matrix, that orders the series in the with aggregated
        first, and bottom levels last.
        - The error covariance matrix, which is used to weight the error in the
        reconciliation.
        """
        self.unique_series_ = _get_unique_series_from_df(X)
        self.S_ = _create_summing_matrix_from_index(self.unique_series_)
        self.A_, self.I_ = _split_summing_matrix(self.S_)

        self._sigma = self.error_covariance_matrix

        if not isinstance(self._sigma, pd.DataFrame):
            self._sigma = self._get_error_covariance_matrix(self._sigma)
        else:
            if not isinstance(self._sigma.index, pd.MultiIndex):
                # Coerce to pd.MultiIndex
                # MultiIndex and Index have a lot of differences
                # that avoid us to use the same code for both
                index = pd.MultiIndex.from_tuples(
                    [
                        x if isinstance(x, tuple) else (x,)
                        for x in self.error_covariance_matrix.index.to_list()
                    ]
                )
                columns = pd.MultiIndex.from_tuples(
                    [
                        x if isinstance(x, tuple) else (x,)
                        for x in self.error_covariance_matrix.columns.to_list()
                    ]
                )
                self._sigma.index = index
                self._sigma.columns = columns
        # If the index has integer values and __total, the
        # sort_index will raise TypeError. We handle this case here.
        # This is a workaround for the issue,
        # and shou

        self._sigma = self._sigma.loc[self.S_.index, self.S_.index]
        self._permutation_matrix = _get_permutation_matrix(self.S_)

    def _transform_reconciler(self, X, y):
        return X

    @property
    def _n_bottom(self):
        return self.S_.shape[1]

    @property
    def _n_not_bottom(self):
        return self._n_series - self._n_bottom

    @property
    def _n_series(self):
        return self.S_.shape[0]

    def _inverse_transform_reconciler(self, X, y=None):
        """
        Apply the optimal reconciliation to the forecasts.

        Uses the constraint matrix approach to reconcile the forecasts.
        """
        X_arr, C, _, E, Pt = self._get_arrays(X)

        E = E + self.alpha * np.eye(E.shape[0])
        # Inverse term
        inv = np.linalg.inv(C @ E @ C.T)
        M = np.eye(self._n_series) - E @ C.T @ inv @ C

        # Expand level 0 with X_arr.shape[0] (timepoints)
        M = np.repeat(M[np.newaxis, :, :], X_arr.shape[0], axis=0)

        # Reconciled forecasts
        Y = Pt @ M @ X_arr
        Y = np.moveaxis(Y, 0, 1).reshape((-1, 1))
        df = pd.DataFrame(
            Y,
            index=X.index,
            columns=X.columns,
        )
        return df

    def _get_error_covariance_matrix(self, error_covariance_matrix):
        if error_covariance_matrix == "ols" or error_covariance_matrix is None:
            values = np.eye(len(self.unique_series_))

        elif error_covariance_matrix == "wls_str":
            diag = self.S_.sum(axis=1).values.flatten()
            values = np.diag(diag)
        else:
            raise ValueError(
                f"Error covariance matrix {error_covariance_matrix} not"
                "recognized. Available options are 'ols', 'wls_str'"
            )

        return pd.DataFrame(
            values,
            index=self.S_.index,
            columns=self.S_.index,
        )

    def _get_arrays(self, X):
        """
        Get the arrays needed for the reconciliation.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.

        Returns
        -------
        X_arr : np.ndarray
            The input data as an ndarray (T, N_SERIES, 1), reordered
            so that aggregated levels come first.
        M : np.ndarray
            The matrix that reconciles the base forecasts
        S : np.ndarray
            The summing matrix, reordered
        E : np.ndarray
            The error covariance matrix, reordered
        Pt : np.ndarray
            The inverse of the permutation matrix
        """
        X = X.sort_index()

        # Convert to ndarray (T, N_SERIES, 1)
        X_arr = _dataframe_to_ndarray(X)
        # Permutation matrix to reorder the series

        # The inverse of the permutation matrix is the transpose
        # of itself, since it is orthogonal
        # Shape is (n_series, n_series)
        P = self._permutation_matrix
        Pt = P.T
        # Now, the shape will be (1, N_SERIES, N_SERIES)
        P = np.repeat(P[np.newaxis, :, :], X_arr.shape[0], axis=0)
        Pt = np.repeat(Pt[np.newaxis, :, :], X_arr.shape[0], axis=0)
        # We reorder the series
        X_arr = P @ X_arr

        # Summing matrix, with the correct order after permutation.
        # We use P[0] instead of P since we have just added a new axis
        S = self.S_.values
        S = P[0] @ S
        # A is the matrix that maps the bottom nodes to the non-bottom nodes.
        # Because of the ordering after permutation, we can just take the first
        # n_not_bottom columns
        A = S[: self._n_not_bottom, :]
        # I_na is the vector that maps the non-bottom nodes to themselves
        I_na = np.eye(self._n_not_bottom)
        # E is the error covariance matrix
        # We have to reorder it as well
        # We use Pt[0] to map to the order of the covariance matrix
        # Then P[0] to map to the order of the series. This is like
        # reordering both columns and rows of the covariance matrix
        E = P[0] @ self._sigma.values @ Pt[0]

        # The constrain matrix
        # Concat C =  [A, I_na]
        C = np.concatenate([I_na, -A], axis=1)
        return X_arr, C, S, E, Pt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"error_covariance_matrix": "ols", "alpha": 0.1},
            {"error_covariance_matrix": "wls_str"},
        ]


class NonNegativeOptimalReconciler(OptimalReconciler):
    """
    Apply non-negative reconciliation to hierarchical time series.

    Uses all the forecasts to obtain a reconciled forecast, avoiding
    negative values.

    The optimization problem tries to find the bottom forecasts $b$ which are
    non-negative and minimize the distance between the base forecasts and the
    reconciled forecasts, given the invertion of error covariance matrix $E$ as
    a weighting matrix.

    Parameters
    ----------
    error_covariance_matrix : pd.DataFrame, default=None
        Error covariance matrix. If None, it is assumed to be
        the identity matrix
    alpha : float, default=0
        Constant added to the diagonal of the inverted matrix.

    Examples
    --------
    >>> from sktime.transformations.hierarchical.reconcile import (
    ...     NonNegativeOptimalReconciler)
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    >>> y = _make_hierarchical()
    >>> pipe = NonNegativeOptimalReconciler() * ExponentialSmoothing()
    >>> pipe = pipe.fit(y)
    >>> y_pred = pipe.predict(fh=[1,2,3])
    """

    _tags = {
        "python_dependencies": ["cvxpy"],
    }

    def _inverse_transform_reconciler(self, X, y=None):
        import cvxpy as cp

        X_arr, _, S, E, Pt = self._get_arrays(X)

        X_opt = np.zeros_like(X_arr)

        # We want to find bottom forecasts b so that
        # we minimize
        # ||X - S[0]b||_E-1^2

        Einv = np.linalg.inv(E)
        for t in range(X_arr.shape[0]):
            x_t = cp.Variable((self._n_bottom, 1), nonneg=True)

            # Base forecast for time t
            x_base_t = X_arr[t]

            # Define the objective
            # If Q is the weighting matrix, use quad_form(x_t - x_base_t, Q).
            objective = cp.Minimize(cp.quad_form(x_base_t - S @ x_t, Einv))

            # Example constraints: x_t >= 0
            # Add your hierarchical constraints if necessary
            constraints = [x_t >= 0]

            # Form and solve the problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, verbose=False)

            # Store the optimized result
            X_opt[t] = S @ x_t.value

        # Reconciled forecasts
        Y = Pt @ X_opt
        Y = np.moveaxis(Y, 0, 1).reshape((-1, 1))
        df = pd.DataFrame(
            Y,
            index=X.index,
            columns=X.columns,
        )
        return df


def _create_summing_matrix_from_index(hier_index):
    """
    Get the summing matrix from a hierarchical index.

    Given a MultiIndex 'hier_index' of a hierarchical time series
    (following an sktime-like convention), return a summation matrix S
    as a DataFrame. Each row corresponds to a node in the hierarchy,
    and each column corresponds to a bottom (leaf) node.
    The entry S[i, j] = 1 if row i (an aggregator node) is an ancestor
    of column j (a bottom node), else 0.

    Parameters
    ----------
    hier_index : pd.MultiIndex
        A hierarchical index.

    Returns
    -------
    S_df : pd.DataFrame
        A DataFrame with the same row index and MultiIndex columns as
        'hier_index'.
        Each entry S_df[i, j] = 1 if row i is an ancestor of column j
    """
    # Convert index to list of tuples for convenience
    all_nodes = list(hier_index)

    # Identify bottom nodes (leaf series): those with no '__total' in their tuple
    bottom_nodes = [node for node in all_nodes if "__total" not in node]

    # Initialize an (N x M) matrix: N = total nodes, M = bottom nodes
    N = len(all_nodes)
    M = len(bottom_nodes)
    S = np.zeros((N, M), dtype=int)

    # Populate the summation matrix
    for i, agg_node in enumerate(all_nodes):
        for j, bottom_node in enumerate(bottom_nodes):
            if _is_ancestor(agg_node, bottom_node):
                S[i, j] = 1

    # Create a DataFrame with the same row index and MultiIndex columns
    index = pd.MultiIndex.from_tuples(all_nodes, names=hier_index.names)
    columns = pd.MultiIndex.from_tuples(bottom_nodes, names=hier_index.names)

    S_df = (
        pd.DataFrame(
            S,
            index=index,
            columns=columns,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    return S_df


def _dataframe_to_ndarray(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a hierarchical DataFrame to a hierarchical ndarray.

    The final array have shape (num_timepoints, num_series, 1).

    """
    df = df.sort_index()

    arr = df.values.reshape(-1, 1)

    num_timepoints = df.index.get_level_values(-1).nunique()

    arr = arr.reshape(-1, num_timepoints, 1)
    arr = np.moveaxis(arr, 0, 1)
    return arr


def _split_summing_matrix(S):
    bottom_nodes = S.columns
    not_bottom_nodes = S.index.difference(bottom_nodes)

    I = S.loc[bottom_nodes, bottom_nodes]
    A = S.loc[not_bottom_nodes, bottom_nodes]

    return A, I


def _get_bottom_and_aggregated_idxs(S):
    bottom_levels = S.index.isin(S.columns)
    agg_levels = ~bottom_levels

    # Turn into permutation matrix of shape (n_levels, n_series)
    bottom_idxs = np.eye(len(S))[bottom_levels]
    agg_idxs = np.eye(len(S))[agg_levels]
    return bottom_idxs, agg_idxs


def _get_permutation_matrix(S):
    bottom_idxs, agg_idxs = _get_bottom_and_aggregated_idxs(S)
    return np.concatenate([agg_idxs, bottom_idxs], axis=0)


def _get_unique_series_from_df(X):
    if X.index.nlevels == 1:
        return pd.MultiIndex.from_tuples([("__total",)])
    return pd.MultiIndex.from_frame(X.index.droplevel(-1).unique().to_frame())
