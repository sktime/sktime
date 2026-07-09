"""Smooth forecast reconciliation."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["sktime developers"]

import numpy as np
import pandas as pd

from sktime.transformations.hierarchical.reconcile._base import _ReconcilerTransformer
from sktime.transformations.hierarchical.reconcile._optimal import (
    _create_summing_matrix_from_index,
    _dataframe_to_ndarray,
    _get_permutation_matrix,
    _get_unique_series_from_df,
    _split_summing_matrix,
)

__all__ = ["SmoothReconciler"]


class SmoothReconciler(_ReconcilerTransformer):
    """
    Smooth forecast reconciliation for hierarchical time series.

    Implements the method from Ando (2024) [1]_, which combines MinT
    reconciliation with the Hodrick-Prescott filter to produce forecasts
    that are both coherent and smooth over the forecast horizon.

    The reconciled forecasts solve:

        min_y  ||y - y_base||^2_{Sigma^-1} + lam * ||D2 y||^2

    subject to the coherency constraint C y = 0, where D2 is the
    second-difference matrix (HP filter penalty).

    The closed-form solution is:

        y_smooth = (I + lam * D2'D2)^{-1} * y_mint

    where y_mint is the standard MinT reconciled forecast.

    If the input is not hierarchical, acts as identity.

    Parameters
    ----------
    lam : float, default=1600
        HP smoothing parameter. Higher values give smoother forecasts.
        1600 is the standard quarterly convention; use 100 for annual
        or 14400 for monthly data.
    error_covariance_matrix : pd.DataFrame, str, or None, default=None
        Error covariance matrix for MinT weighting.
        None or "ols" uses the identity (OLS). "wls_str" uses a diagonal
        matrix proportional to the number of bottom-level series aggregated.
        A pd.DataFrame uses the supplied matrix directly.
    alpha : float, default=0
        Regularization added to the diagonal before inversion.

    Examples
    --------
    >>> from sktime.transformations.hierarchical.reconcile import SmoothReconciler
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = _make_hierarchical()
    >>> pipe = SmoothReconciler(lam=1600) * NaiveForecaster()
    >>> pipe = pipe.fit(y)
    >>> y_pred = pipe.predict(fh=[1, 2, 3])

    References
    ----------
    .. [1] Ando, S. (2024). Smooth Forecast Reconciliation. IMF Working Paper
       WP/24/57. https://www.imf.org/en/Publications/WP/Issues/2024/03/22/
       Smooth-Forecast-Reconciliation-546654
    """

    _tags = {
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "tests:core": True,
    }

    def __init__(self, lam=1600, error_covariance_matrix=None, alpha=0):
        self.lam = lam
        self.error_covariance_matrix = error_covariance_matrix
        self.alpha = alpha
        super().__init__()

    def _fit_reconciler(self, X, y=None):
        self.unique_series_ = _get_unique_series_from_df(X)
        self.S_ = _create_summing_matrix_from_index(self.unique_series_)
        self.A_, self.I_ = _split_summing_matrix(self.S_)
        self._permutation_matrix = _get_permutation_matrix(self.S_)

        sigma = self.error_covariance_matrix
        if not isinstance(sigma, pd.DataFrame):
            self._sigma = self._get_error_covariance_matrix(sigma)
        else:
            if not isinstance(sigma.index, pd.MultiIndex):
                index = pd.MultiIndex.from_tuples(
                    [x if isinstance(x, tuple) else (x,) for x in sigma.index.tolist()]
                )
                columns = pd.MultiIndex.from_tuples(
                    [
                        x if isinstance(x, tuple) else (x,)
                        for x in sigma.columns.tolist()
                    ]
                )
                sigma = sigma.copy()
                sigma.index = index
                sigma.columns = columns
            self._sigma = sigma.loc[self.S_.index, self.S_.index]

        return self

    def _transform_reconciler(self, X, y=None):
        return X

    def _inverse_transform_reconciler(self, X, y=None):
        X = X.sort_index()

        n_timepoints = X.index.get_level_values(-1).nunique()
        n_series = len(self.unique_series_)

        X_arr = _dataframe_to_ndarray(X)

        P = self._permutation_matrix
        Pt = P.T
        P_rep = np.repeat(P[np.newaxis, :, :], n_timepoints, axis=0)
        Pt_rep = np.repeat(Pt[np.newaxis, :, :], n_timepoints, axis=0)
        X_perm = P_rep @ X_arr

        S_perm = P @ self.S_.values
        n_not_bottom = n_series - self.S_.shape[1]
        A = S_perm[:n_not_bottom, :]
        C = np.concatenate([np.eye(n_not_bottom), -A], axis=1)

        E = P @ self._sigma.values @ Pt + self.alpha * np.eye(n_series)

        inv_mint = np.linalg.inv(C @ E @ C.T)
        M_mint = np.eye(n_series) - E @ C.T @ inv_mint @ C
        Y_mint = M_mint @ X_perm

        if n_timepoints >= 3:
            D2 = _build_second_difference_matrix(n_timepoints)
            H_inv = np.linalg.inv(np.eye(n_timepoints) + self.lam * (D2.T @ D2))
            Y = Y_mint[:, :, 0]
            Y_smooth = (H_inv @ Y)[:, :, np.newaxis]
        else:
            Y_smooth = Y_mint

        Y_out = np.moveaxis(Pt_rep @ Y_smooth, 0, 1).reshape((-1, 1))
        return pd.DataFrame(Y_out, index=X.index, columns=X.columns)

    def _get_error_covariance_matrix(self, spec):
        if spec is None or spec == "ols":
            values = np.eye(len(self.unique_series_))
        elif spec == "wls_str":
            values = np.diag(self.S_.sum(axis=1).values.flatten())
        else:
            raise ValueError(
                f"error_covariance_matrix '{spec}' not recognized. "
                "Use None, 'ols', or 'wls_str'."
            )
        return pd.DataFrame(values, index=self.S_.index, columns=self.S_.index)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"lam": 1600, "error_covariance_matrix": None},
            {"lam": 100, "error_covariance_matrix": "wls_str", "alpha": 0.1},
        ]


def _build_second_difference_matrix(T):
    """Build the second-difference matrix of shape (T-2, T)."""
    if T < 3:
        return np.zeros((0, T))
    D2 = np.zeros((T - 2, T))
    for i in range(T - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1
    return D2
