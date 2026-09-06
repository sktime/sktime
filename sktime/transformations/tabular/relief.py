"""Feature selection algorithms: mRMR, ReliefF and RReliefF.

This module implements three powerful feature selection algorithms:
- mRMR (Minimum Redundancy Maximum Relevance): Selects features that have high
  relevance with the target and low redundancy among themselves.
- ReliefF: A multiclass feature selection method that weighs features based on
  their ability to distinguish between instances of different classes.
- RReliefF: A regression variant of ReliefF that handles continuous target variables
  by using intermediate weights to compute feature importance.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Manas-7854"]
__all__ = ["mRMR", "ReliefFTransformer", "RReliefFTransformer"]

import numbers

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class mRMR(BaseEstimator, TransformerMixin):
    """Minimum Redundancy Maximum Relevance (mRMR) feature selector.

    The mRMR algorithm finds an optimal set of features that is mutually and
    maximally dissimilar and can represent the response variable effectively.
    The algorithm minimizes the redundancy of a feature set and maximizes the
    relevance of a feature set to the response variable using mutual information.

    Parameters
    ----------
    n_features : int, default=10
        Number of features to select.

    task : str, default='auto'
        Type of task. Options: 'classification', 'regression', 'auto'.
        If 'auto', will be inferred from the target variable.

    n_bins : int, default=10
        Number of bins to use when discretizing continuous features for
        mutual information calculation.

    random_state : int, RandomState instance or None, default=None
        Controls randomization for mutual information estimation and
        for breaking ties in feature selection.

    Attributes
    ----------
    selected_features_ : ndarray of shape (n_features,)
        Indices of selected features.

    scores_ : ndarray of shape (n_features_in_,)
        Feature importance scores. Higher scores indicate more important features.

    relevances_ : ndarray of shape (n_features_in_,)
        Relevance scores (mutual information with target) for each feature.

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    >>> selector = mRMR(n_features=5)
    >>> X_selected = selector.fit_transform(X, y)
    >>> X_selected.shape
    (100, 5)
    """

    def __init__(self, n_features=10, task="auto", n_bins=10, random_state=None):
        self.n_features = n_features
        self.task = task
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the mRMR feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate inputs
        X, y = check_X_y(X, y, dtype=np.float64)

        # Store attributes
        self.n_features_in_ = X.shape[1]
        n_samples, n_total_features = X.shape

        # Determine task type
        if self.task == "auto":
            if len(np.unique(y)) <= 10 and np.all(y == y.astype(int)):
                task_type = "classification"
            else:
                task_type = "regression"
        else:
            task_type = self.task

        # Limit n_features to available features
        n_features_to_select = min(self.n_features, n_total_features)

        # Calculate relevances (mutual information with target)
        if task_type == "classification":
            self.relevances_ = mutual_info_classif(
                X, y, discrete_features=False, random_state=self.random_state
            )
        else:
            self.relevances_ = mutual_info_regression(
                X, y, discrete_features=False, random_state=self.random_state
            )

        # Initialize
        selected_indices = []
        remaining_indices = list(range(n_total_features))
        scores = np.zeros(n_total_features)

        # Step 1: Select feature with largest relevance
        if len(remaining_indices) > 0:
            relevances_remaining = self.relevances_[remaining_indices]
            if np.max(relevances_remaining) > 0:
                best_idx_pos = np.argmax(relevances_remaining)
                best_idx = remaining_indices[best_idx_pos]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                scores[best_idx] = self.relevances_[best_idx]

        # Steps 2-3: Select features with nonzero relevance and zero redundancy
        while (
            len(selected_indices) < n_features_to_select and len(remaining_indices) > 0
        ):
            # Calculate redundancies for remaining features
            zero_redundancy_candidates = []

            for idx in remaining_indices:
                if self.relevances_[idx] > 0:
                    # Calculate redundancy with already selected features
                    redundancy = 0
                    if len(selected_indices) > 0:
                        redundancy = self._calculate_redundancy(
                            X, idx, selected_indices
                        )

                    if redundancy == 0:
                        zero_redundancy_candidates.append((idx, self.relevances_[idx]))

            # If we found zero redundancy candidates, select best one
            if zero_redundancy_candidates:
                best_candidate = max(zero_redundancy_candidates, key=lambda x: x[1])
                best_idx = best_candidate[0]
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                scores[best_idx] = self.relevances_[best_idx]
            else:
                break

        # Steps 4-5: Select features using MIQ (relevance/redundancy ratio)
        while (
            len(selected_indices) < n_features_to_select and len(remaining_indices) > 0
        ):
            best_miq = -np.inf
            best_idx = None

            for idx in remaining_indices:
                if self.relevances_[idx] > 0:
                    redundancy = self._calculate_redundancy(X, idx, selected_indices)

                    if redundancy > 0:
                        miq = self.relevances_[idx] / redundancy
                        if miq > best_miq:
                            best_miq = miq
                            best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                scores[best_idx] = best_miq
            else:
                break

        # Step 6: Add remaining features with zero relevance in random order
        zero_relevance_features = [
            idx for idx in remaining_indices if self.relevances_[idx] == 0
        ]

        if zero_relevance_features and len(selected_indices) < n_features_to_select:
            np.random.seed(self.random_state)
            np.random.shuffle(zero_relevance_features)

            n_to_add = min(
                len(zero_relevance_features),
                n_features_to_select - len(selected_indices),
            )

            for i in range(n_to_add):
                idx = zero_relevance_features[i]
                selected_indices.append(idx)
                scores[idx] = 0

        # Store results
        self.selected_features_ = np.array(selected_indices[:n_features_to_select])
        self.scores_ = scores

        return self

    def transform(self, X):
        """Transform X by selecting features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_selected_features)
            Transformed data with only selected features.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this "
                f"selector was fitted with {self.n_features_in_} features."
            )

        return X[:, self.selected_features_]

    def _calculate_redundancy(self, X, feature_idx, selected_indices):
        """Calculate average redundancy of a feature with selected features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        feature_idx : int
            Index of the feature to calculate redundancy for.
        selected_indices : list
            Indices of already selected features.

        Returns
        -------
        redundancy : float
            Average mutual information between the feature and selected features.
        """
        if len(selected_indices) == 0:
            return 0.0

        redundancies = []
        feature_data = X[:, feature_idx].reshape(-1, 1)

        for selected_idx in selected_indices:
            selected_data = X[:, selected_idx]

            # Calculate mutual information between features
            # We discretize both features for MI calculation
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins, encode="ordinal", strategy="uniform", subsample=None
            )

            try:
                feature_discrete = discretizer.fit_transform(feature_data).ravel()
                selected_discrete = discretizer.fit_transform(
                    selected_data.reshape(-1, 1)
                ).ravel()

                # Calculate mutual information
                mi = self._mutual_info_discrete(feature_discrete, selected_discrete)
                redundancies.append(mi)

            except Exception:
                # Fallback: use correlation-based measure if MI fails
                corr = np.abs(np.corrcoef(X[:, feature_idx], X[:, selected_idx])[0, 1])
                redundancies.append(corr if not np.isnan(corr) else 0.0)

        return np.mean(redundancies)

    def _mutual_info_discrete(self, x, y):
        """Calculate mutual information between two discrete variables.

        Parameters
        ----------
        x, y : array-like
            Discrete variables.

        Returns
        -------
        mi : float
            Mutual information.
        """
        # Create contingency table
        unique_x = np.unique(x)
        unique_y = np.unique(y)

        contingency = np.zeros((len(unique_x), len(unique_y)))

        for i, val_x in enumerate(unique_x):
            for j, val_y in enumerate(unique_y):
                contingency[i, j] = np.sum((x == val_x) & (y == val_y))

        # Add small constant to avoid log(0)
        contingency = contingency + 1e-10

        # Calculate probabilities
        p_xy = contingency / np.sum(contingency)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # Calculate mutual information
        mi = 0.0
        for i in range(len(unique_x)):
            for j in range(len(unique_y)):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

        return max(0.0, mi)  # Ensure non-negative

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : ndarray of str
            Selected feature names.
        """
        check_is_fitted(self)

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        return np.array(
            [input_features[i] for i in self.selected_features_], dtype=object
        )

    def _more_tags(self):
        """Return additional tags for the estimator."""
        return {
            "requires_y": True,
            "requires_fit": True,
            "X_types": ["2darray"],
            "y_types": ["1dlabels"],
            "no_validation": False,
        }


class ReliefFTransformer(TransformerMixin, BaseEstimator):
    """ReliefF feature selector for multiclass classification.

    ReliefF finds the weights of predictors in the case where `y` is a multiclass
    categorical variable. The algorithm penalizes the predictors that give different
    values to neighbors of the same class, and rewards predictors that give different
    values to neighbors of different classes.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        The number of features to select. If None, all features are kept.
        If int, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

    k : int, default=10
        Number of nearest neighbors to consider for each instance.

    updates : int, default=100
        Number of iterations (random instances) to process.

    sigma : float, default=1.0
        Scaling parameter for distance weighting in the exponential function.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=str
        Names of features seen during fit. Defined only when X has feature
        names that are all strings.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Feature importance scores computed by ReliefF algorithm.

    selected_features_ : ndarray of shape (n_features_to_select,)
        Indices of selected features.

    support_ : ndarray of shape (n_features_in_,), dtype=bool
        The mask of selected features.
    """

    _parameter_constraints = {
        "n_features_to_select": [
            None,
            "int32",
            Interval(numbers.Real, 0.0, 1.0, closed="neither"),
        ],
        "k": [Interval(numbers.Integral, 1, None, closed="left")],
        "updates": [Interval(numbers.Integral, 1, None, closed="left")],
        "sigma": [Interval(numbers.Real, 0.0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(
        self, n_features=None, k=10, updates=100, sigma=1.0, random_state=None
    ):
        self.n_features_to_select = n_features
        self.k = k
        self.updates = updates
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the ReliefF feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input data
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        # Store input validation attributes
        self.n_features_in_ = X.shape[1]
        self._check_feature_names(X, reset=True)

        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)

        # Compute feature importances using ReliefF algorithm
        self.feature_importances_ = self._compute_relieff_weights(X, y_encoded)

        # Determine number of features to select
        n_features_to_select = self._get_n_features_to_select()

        # Select top features based on importance scores
        self.selected_features_ = np.argsort(self.feature_importances_)[::-1][
            :n_features_to_select
        ]

        # Create support mask
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_features_] = True

        return self

    def _compute_relieff_weights(self, X, y):
        """Compute ReliefF feature weights."""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)

        # Get unique classes and their prior probabilities
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_probs = class_counts / n_samples
        class_prob_dict = dict(zip(unique_classes, class_probs))

        # Random state for reproducibility
        rng = check_random_state(self.random_state)

        # Precompute feature ranges for continuous features
        feature_ranges = np.ptp(X, axis=0)
        # Avoid division by zero
        feature_ranges[feature_ranges == 0] = 1.0

        # Main ReliefF iterations
        for iteration in range(self.updates):
            # Randomly select an instance
            r_idx = rng.randint(0, n_samples)
            x_r = X[r_idx]
            y_r = y[r_idx]

            # For each class, find k nearest neighbors
            for target_class in unique_classes:
                class_indices = np.where(y == target_class)[0]

                if len(class_indices) == 0:
                    continue

                # Find k nearest neighbors in this class
                if len(class_indices) == 1 and target_class == y_r:
                    # If only one instance in class and it's the selected instance
                    continue

                # Remove the selected instance if it's in this class
                if target_class == y_r and len(class_indices) > 1:
                    class_indices = class_indices[class_indices != r_idx]

                if len(class_indices) == 0:
                    continue

                # Compute distances to all instances in this class
                distances = np.linalg.norm(X[class_indices] - x_r, axis=1)

                # Get k nearest neighbors (or all if less than k available)
                k_actual = min(self.k, len(class_indices))
                nearest_indices = np.argsort(distances)[:k_actual]

                # Update weights for each nearest neighbor
                for rank, nn_idx in enumerate(nearest_indices):
                    q_idx = class_indices[nn_idx]
                    x_q = X[q_idx]
                    y_q = y[q_idx]

                    # Compute scaled distance weight
                    d_rq_scaled = np.exp(-(((rank + 1) / self.sigma) ** 2))

                    # Normalize by sum of all scaled distances in this class
                    sum_scaled_distances = np.sum(
                        [
                            np.exp(-(((i + 1) / self.sigma) ** 2))
                            for i in range(k_actual)
                        ]
                    )
                    d_rq = d_rq_scaled / sum_scaled_distances

                    # Compute feature differences
                    delta_j = self._compute_feature_differences(
                        x_r, x_q, feature_ranges
                    )

                    # Update weights based on instances (same or different class)
                    if y_r == y_q:  # Same class
                        weights -= (delta_j / self.updates) * d_rq
                    else:  # Different class
                        p_yr = class_prob_dict[y_r]
                        p_yq = class_prob_dict[y_q]
                        weight_factor = p_yq / (1 - p_yr) if p_yr != 1.0 else 0.0
                        weights += weight_factor * (delta_j / self.updates) * d_rq

        return weights

    def _compute_feature_differences(self, x_r, x_q, feature_ranges):
        """Compute feature differences between two instances."""
        # For continuous features, use normalized absolute difference
        delta_j = np.abs(x_r - x_q) / feature_ranges
        return delta_j

    def _get_n_features_to_select(self):
        """Determine the number of features to select."""
        if self.n_features_to_select is None:
            return self.n_features_in_
        elif isinstance(self.n_features_to_select, float):
            return int(self.n_features_to_select * self.n_features_in_)
        else:
            return min(self.n_features_to_select, self.n_features_in_)

    def transform(self, X):
        """Transform X by selecting features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_to_select)
            Transformed data with selected features only.
        """
        # Check if fitted
        if not hasattr(self, "support_"):
            raise ValueError(
                "This ReliefFTransformer instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        # Validate input
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # Check feature names if they were provided during fit
        self._check_feature_names(X, reset=False)

        # Check that X has the same number of features as during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but ReliefFTransformer "
                f"is expecting {self.n_features_in_} features as seen in fit."
            )

        # Select features
        return X[:, self.support_]

    def _check_feature_names(self, X, *, reset):
        """Check feature names."""
        if hasattr(X, "columns"):
            feature_names_in = np.array(X.columns, dtype=object)
        else:
            feature_names_in = None

        if reset:
            self.feature_names_in_ = feature_names_in
        else:
            if hasattr(self, "feature_names_in_"):
                if feature_names_in is not None and self.feature_names_in_ is not None:
                    if not np.array_equal(feature_names_in, self.feature_names_in_):
                        raise ValueError(
                            "Feature names are different from those seen in fit."
                        )
                elif (feature_names_in is None) != (self.feature_names_in_ is None):
                    raise ValueError(
                        "X has different feature names than those seen in fit."
                    )

    def get_support(self, indices=False):
        """Get a mask, or integer index, of the selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        if not hasattr(self, "support_"):
            raise ValueError("This ReliefFTransformer instance is not fitted yet.")

        if indices:
            return self.selected_features_
        else:
            return self.support_

    def _more_tags(self):
        """Return additional tags for the estimator."""
        return {
            "requires_y": True,
            "X_types": ["2darray"],
            "allow_nan": False,
            "requires_fit": True,
        }


class RReliefFTransformer(TransformerMixin, BaseEstimator):
    """RReliefF feature selector for continuous regression targets.

    RReliefF works with continuous `y`. Similar to ReliefF, RReliefF also penalizes
    the predictors that give different values to neighbors with the same response
    values, and rewards predictors that give different values to neighbors with
    different response values. However, RReliefF uses intermediate weights to
    compute the final predictor weights.

    Parameters
    ----------
    n_features_to_select : int or float, default=None
        The number of features to select. If None, all features are kept.
        If int, the parameter is the absolute number of features to select.
        If float between 0 and 1, it is the fraction of features to select.

    k : int, default=10
        Number of nearest neighbors to consider for each instance.

    updates : int, default=100
        Number of iterations (random instances) to process.

    sigma : float, default=1.0
        Scaling parameter for distance weighting in the exponential function.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=str
        Names of features seen during fit. Defined only when X has feature
        names that are all strings.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Feature importance scores computed by RReliefF algorithm.

    selected_features_ : ndarray of shape (n_features_to_select,)
        Indices of selected features.

    support_ : ndarray of shape (n_features_in_,), dtype=bool
        The mask of selected features.

    intermediate_weights_ : dict
        Dictionary containing the intermediate weights computed during training:
        - 'W_dy': Weight of having different response values
        - 'W_dj': Weights of having different predictor values
        - 'W_dy_and_dj': Weights of having different response and predictor values
    """

    _parameter_constraints = {
        "n_features_to_select": [
            None,
            "int32",
            Interval(numbers.Real, 0.0, 1.0, closed="neither"),
        ],
        "k": [Interval(numbers.Integral, 1, None, closed="left")],
        "updates": [Interval(numbers.Integral, 1, None, closed="left")],
        "sigma": [Interval(numbers.Real, 0.0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(
        self, n_features=None, k=10, updates=100, sigma=1.0, random_state=None
    ):
        self.n_features_to_select = n_features
        self.k = k
        self.updates = updates
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the RReliefF feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (continuous).

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input data
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)

        # Store input validation attributes
        self.n_features_in_ = X.shape[1]
        self._check_feature_names(X, reset=True)

        # Compute feature importances using RReliefF algorithm
        self.feature_importances_, self.intermediate_weights_ = (
            self._compute_rrelieff_weights(X, y)
        )

        # Determine number of features to select
        n_features_to_select = self._get_n_features_to_select()

        # Select top features based on importance scores
        self.selected_features_ = np.argsort(self.feature_importances_)[::-1][
            :n_features_to_select
        ]

        # Create support mask
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_features_] = True

        return self

    def _compute_rrelieff_weights(self, X, y):
        """Compute RReliefF feature weights using intermediate weights."""
        n_samples, n_features = X.shape

        # Initialize intermediate weights
        W_dy = 0.0  # Weight of having different response values
        W_dj = np.zeros(n_features)  # Weight of having different predictor values
        W_dy_and_dj = np.zeros(
            n_features
        )  # Weight of having different response and predictor values

        # Random state for reproducibility
        rng = check_random_state(self.random_state)

        # Precompute feature ranges for continuous features
        feature_ranges = np.ptp(X, axis=0)
        # Avoid division by zero
        feature_ranges[feature_ranges == 0] = 1.0

        # Precompute y range for response difference computation
        y_range = np.ptp(y)
        if y_range == 0:
            y_range = 1.0  # Avoid division by zero if all y values are the same

        # Main RReliefF iterations
        for iteration in range(self.updates):
            # Randomly select an instance
            r_idx = rng.randint(0, n_samples)
            x_r = X[r_idx]
            y_r = y[r_idx]

            # Find k nearest neighbors to x_r
            # Exclude the selected instance itself
            other_indices = np.arange(n_samples)
            other_indices = other_indices[other_indices != r_idx]

            if len(other_indices) == 0:
                continue

            # Compute distances to all other instances
            distances = np.linalg.norm(X[other_indices] - x_r, axis=1)

            # Get k nearest neighbors (or all if less than k available)
            k_actual = min(self.k, len(other_indices))
            nearest_indices = np.argsort(distances)[:k_actual]

            # Update intermediate weights for each nearest neighbor
            for rank, nn_idx in enumerate(nearest_indices):
                q_idx = other_indices[nn_idx]
                x_q = X[q_idx]
                y_q = y[q_idx]

                # Compute scaled distance weight
                d_rq_scaled = np.exp(-(((rank + 1) / self.sigma) ** 2))

                # Normalize by sum of all scaled distances
                sum_scaled_distances = np.sum(
                    [np.exp(-(((i + 1) / self.sigma) ** 2)) for i in range(k_actual)]
                )
                d_rq = d_rq_scaled / sum_scaled_distances

                # Compute response difference
                delta_y = self._compute_response_difference(y_r, y_q, y_range)

                # Compute feature differences
                delta_j = self._compute_feature_differences(x_r, x_q, feature_ranges)

                # Update intermediate weights
                W_dy += delta_y * d_rq
                W_dj += delta_j * d_rq
                W_dy_and_dj += delta_y * delta_j * d_rq

        # Store intermediate weights for inspection
        intermediate_weights = {
            "W_dy": W_dy,
            "W_dj": W_dj.copy(),
            "W_dy_and_dj": W_dy_and_dj.copy(),
        }

        # Compute final predictor weights using the RReliefF formula:
        # W_j = (W_dy_and_dj / W_dy) - (W_dj - W_dy_and_dj) / (m - W_dy)
        final_weights = np.zeros(n_features)

        for j in range(n_features):
            if W_dy > 0:
                term1 = W_dy_and_dj[j] / W_dy
            else:
                term1 = 0.0

            if self.updates - W_dy > 0:
                term2 = (W_dj[j] - W_dy_and_dj[j]) / (self.updates - W_dy)
            else:
                term2 = 0.0

            final_weights[j] = term1 - term2

        return final_weights, intermediate_weights

    def _compute_response_difference(self, y_r, y_q, y_range):
        """Compute response difference between two instances."""
        return np.abs(y_r - y_q) / y_range

    def _compute_feature_differences(self, x_r, x_q, feature_ranges):
        """Compute feature differences between two instances."""
        # For continuous features, use normalized absolute difference
        delta_j = np.abs(x_r - x_q) / feature_ranges
        return delta_j

    def _get_n_features_to_select(self):
        """Determine the number of features to select."""
        if self.n_features_to_select is None:
            return self.n_features_in_
        elif isinstance(self.n_features_to_select, float):
            return int(self.n_features_to_select * self.n_features_in_)
        else:
            return min(self.n_features_to_select, self.n_features_in_)

    def transform(self, X):
        """Transform X by selecting features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_to_select)
            Transformed data with selected features only.
        """
        # Check if fitted
        if not hasattr(self, "support_"):
            raise ValueError(
                "This RReliefFTransformer instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        # Validate input
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # Check feature names if they were provided during fit
        self._check_feature_names(X, reset=False)

        # Check that X has the same number of features as during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but RReliefFTransformer "
                f"is expecting {self.n_features_in_} features as seen in fit."
            )

        # Select features
        return X[:, self.support_]

    def _check_feature_names(self, X, *, reset):
        """Check feature names."""
        if hasattr(X, "columns"):
            feature_names_in = np.array(X.columns, dtype=object)
        else:
            feature_names_in = None

        if reset:
            self.feature_names_in_ = feature_names_in
        else:
            if hasattr(self, "feature_names_in_"):
                if feature_names_in is not None and self.feature_names_in_ is not None:
                    if not np.array_equal(feature_names_in, self.feature_names_in_):
                        raise ValueError(
                            "Feature names are different from those seen in fit."
                        )
                elif (feature_names_in is None) != (self.feature_names_in_ is None):
                    raise ValueError(
                        "X has different feature names than those seen in fit."
                    )

    def get_support(self, indices=False):
        """Get a mask, or integer index, of the selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        if not hasattr(self, "support_"):
            raise ValueError("This RReliefFTransformer instance is not fitted yet.")

        if indices:
            return self.selected_features_
        else:
            return self.support_

    def get_intermediate_weights(self):
        """Get the intermediate weights computed during training.

        Returns
        -------
        intermediate_weights : dict
            Dictionary containing:
            - 'W_dy': Weight of having different response values
            - 'W_dj': Weights of having different predictor values
            - 'W_dy_and_dj': Weights of having different response and predictor values
        """
        if not hasattr(self, "intermediate_weights_"):
            raise ValueError("This RReliefFTransformer instance is not fitted yet.")
        return self.intermediate_weights_.copy()

    def _more_tags(self):
        """Return additional tags for the estimator."""
        return {
            "requires_y": True,
            "X_types": ["2darray"],
            "allow_nan": False,
            "requires_fit": True,
            "y_types": ["continuous"],
        }
