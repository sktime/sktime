"""Transfer Entropy calculation matrix between time series."""

__authors__ = ["Spinachboul"]

__all__ = ["TransferEntropy"]

import numpy as np

from sktime.dists_kernels.base import BasePairwiseTransformerPanel


class TransferEntropy(BasePairwiseTransformerPanel):
    """
    Transfer Entropy-based pairwise distance for panel data.

    Parameters
    ----------
    lag_source : int, default=1
        Number of lagged timesteps for the source time series.
    lag_target : int, default=1
        Number of lagged timesteps for the target time series.
    estimator : str, default="binning"
        Method to estimate probability distributions ("binning" supported).
    n_bins : int, default=10
        Number of bins used if estimator="binning".
    significance_test : bool, default=False
        Whether to perform statistical testing with surrogate data.
    """

    _tags = {
        "symmetric": False,  # TE is directional: TE(X -> Y) != TE(Y -> X)
        "X_inner_mtype": "df-list",
        "pwtrafo_type": "distance",  # could also be "similarity" if TE is flipped
    }

    def __init__(
        self,
        lag_source=1,
        lag_target=1,
        estimator="binning",
        n_bins=10,
        significance_test=False,
    ):
        self.lag_source = lag_source
        self.lag_target = lag_target
        self.estimator = estimator
        self.n_bins = n_bins
        self.significance_test = significance_test

        super().__init__()

    def _transform(self, X, X2=None):
        """Compute the pairwise Transfer Entropy matrix."""
        n = len(X)
        m = len(X2) if X2 is not None else n

        if X2 is None:
            X2 = X

        te_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                X_i = X[i]
                X_j = X2[j]
                te_matrix[i, j] = self._get_transfer_entropy(X_i, X_j)

        return te_matrix

    def _get_transfer_entropy(self, X_src, X_tgt):
        """
        Core function to compute Transfer Entropy from X_src to X_tgt.

        Parameters
        ----------
        X_src : pd.DataFrame
            Source time series.
        X_tgt : pd.DataFrame
            Target time series.

        Returns
        -------
        te_value : float
            Estimated Transfer Entropy from X_src -> X_tgt.
        """
        # Example: flatten all series to 1D for simplicity (expand later)
        x = X_src.values.flatten()
        y = X_tgt.values.flatten()

        # Basic input checking
        min_length = max(self.lag_source, self.lag_target) + 1
        if len(x) < min_length or len(y) < min_length:
            return np.nan

        # Create lagged vectors
        x_lagged = np.array([x[i - self.lag_source] for i in range(min_length, len(x))])
        y_lagged = np.array([y[i - self.lag_target] for i in range(min_length, len(y))])
        y_future = np.array([y[i] for i in range(min_length, len(y))])

        # Check if any of the vectors are empty
        if len(x_lagged) == 0 or len(y_lagged) == 0 or len(y_future) == 0:
            return np.nan

        # Binning (basic estimation)
        if self.estimator == "binning":
            x_lagged_binned = np.digitize(
                x_lagged, np.histogram_bin_edges(x, bins=self.n_bins)
            )
            y_lagged_binned = np.digitize(
                y_lagged, np.histogram_bin_edges(y, bins=self.n_bins)
            )
            y_future_binned = np.digitize(
                y_future, np.histogram_bin_edges(y, bins=self.n_bins)
            )

            # Joint histograms
            p_yfuture_ylag = self._joint_prob(y_future_binned, y_lagged_binned)
            p_yfuture_ylag_xlag = self._joint_prob_3d(
                y_future_binned, y_lagged_binned, x_lagged_binned
            )

            # Marginals
            p_ylag = self._marginal_prob(y_lagged_binned)
            p_ylag_xlag = self._joint_prob(y_lagged_binned, x_lagged_binned)

            # Calculate TE based on conditional entropies
            H1 = self._conditional_entropy(p_yfuture_ylag, p_ylag)
            H2 = self._conditional_entropy(p_yfuture_ylag_xlag, p_ylag_xlag)
            te_value = H1 - H2
            return te_value

        else:
            raise NotImplementedError(
                f"Estimator {self.estimator} not implemented yet."
            )

    # Helper functions
    def _joint_prob(self, x, y):
        """Estimate 2D joint probability."""
        if len(x) == 0 or len(y) == 0:
            return np.array([[0.0]])

        # Add 1 to ensure bins include max value
        bins = (max(1, np.max(x) + 1), max(1, np.max(y) + 1))
        hist, _, _ = np.histogram2d(x, y, bins=bins)
        return hist / np.sum(hist) if np.sum(hist) > 0 else hist

    def _joint_prob_3d(self, x, y, z):
        """Estimate 3D joint probability by flattening triples."""
        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            return np.array([[[0.0]]])

        xyz = np.vstack((x, y, z)).T
        unique, counts = np.unique(xyz, axis=0, return_counts=True)

        # Add 1 to ensure bins include max value
        max_x = max(1, np.max(x) + 1) if len(x) > 0 else 1
        max_y = max(1, np.max(y) + 1) if len(y) > 0 else 1
        max_z = max(1, np.max(z) + 1) if len(z) > 0 else 1

        hist = np.zeros((max_x, max_y, max_z))
        for (i, j, k), c in zip(unique, counts):
            hist[i, j, k] = c
        return hist / np.sum(hist) if np.sum(hist) > 0 else hist

    def _marginal_prob(self, x):
        """Estimate 1D marginal probability."""
        if len(x) == 0:
            return np.array([0.0])

        # Add 1 to ensure bins include max value
        bins = max(1, np.max(x) + 1)
        hist, _ = np.histogram(x, bins=bins)
        return hist / np.sum(hist) if np.sum(hist) > 0 else hist

    def _conditional_entropy(self, joint, marginal):
        """Calculate conditional entropy H(Y|X) = H(X,Y) - H(X)."""
        eps = 1e-10
        # Skip calculation if arrays are all zeros
        if np.sum(joint) == 0 or np.sum(marginal) == 0:
            return 0.0

        joint = np.clip(joint, eps, 1.0)
        marginal = np.clip(marginal, eps, 1.0)
        H_joint = -np.sum(joint * np.log(joint))
        H_marginal = -np.sum(marginal * np.log(marginal))
        return H_joint - H_marginal

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return default test parameters."""
        params1 = {
            "lag_source": 1,
            "lag_target": 1,
            "estimator": "binning",
            "n_bins": 10,
            "significance_test": False,
        }
        params2 = {
            "lag_source": 2,
            "lag_target": 3,
            "estimator": "binning",
            "n_bins": 5,
            "significance_test": True,
        }

        return [params1, params2]
