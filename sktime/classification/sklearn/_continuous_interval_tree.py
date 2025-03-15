"""Continuous interval tree (CIT) vector classifier (aka Time Series Tree).

Continuous Interval Tree aka Time Series Tree, base classifier originally used in the
time series forest interval based classification algorithm. Fits sklearn conventions.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["ContinuousIntervalTree"]

import sys

import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from sktime.exceptions import NotFittedError


class ContinuousIntervalTree(BaseEstimator):
    """Continuous interval tree (CIT) vector classifier (aka Time Series Tree).

    The ``Time Series Tree`` described in the Time Series Forest (TSF) paper Deng et al
    (2013) [1]. A simple information gain based tree for continuous attributes using a
    bespoke margin gain metric for tie breaking.

    Implemented as a bade classifier for interval based time series classifiers such as
    ``CanonicalIntervalForest`` and ``DrCIF``.

    Parameters
    ----------
    max_depth : int, default=sys.maxsize
        Maximum depth for the tree.
    thresholds : int, default=20
        Number of thresholds to split continuous attributes on at tree nodes.
    random_state : int, RandomState instance or None, default=None
        If ``int``, random_state is the seed used by the random number generator;
        If ``RandomState`` instance, random_state is the random number generator;
        If ``None``, the random number generator is the ``RandomState`` instance used
        by ``np.random``.

    Attributes
    ----------
    classes_ : list
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    n_instances_ : int
        The number of train cases in the training set.
    n_atts_ : int
        The number of attributes in the training set.

    See Also
    --------
    CanonicalIntervalForest
    DrCIF

    Notes
    -----
    For the Java version, see
    `tsml <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    machine_learning/classifiers/ContinuousIntervalTree.java>`_.

    References
    ----------
    .. [1] H.Deng, G.Runger, E.Tuv and M.Vladimir, "A time series forest for
       classification and feature extraction",Information Sciences, 239, 2013

    Examples
    --------
    >>> from sktime.classification.sklearn import ContinuousIntervalTree
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> X_train = from_nested_to_3d_numpy(X_train)
    >>> X_test = from_nested_to_3d_numpy(X_test)
    >>> clf = ContinuousIntervalTree()
    >>> clf.fit(X_train, y_train)
    ContinuousIntervalTree(...)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        max_depth=sys.maxsize,
        thresholds=20,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.thresholds = thresholds
        self.random_state = random_state

        super().__init__()

    def fit(self, X, y):
        """Fit a tree on cases (X,y), where y is the target variable.

        Build an information gain based tree for continuous attributes using the
        margin gain metric for ties.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_instances, n_attributes]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        from sktime.classification.sklearn._continuous_interval_tree_numba import (
            _entropy,
        )

        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "ContinuousIntervalTree is not a time series classifier. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, force_all_finite="allow-nan"
        )

        self.n_instances_, self.n_atts_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._class_dictionary = {}
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        # escape if only one class seen
        if self.n_classes_ == 1:
            self._is_fitted = True
            return self

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        rng = check_random_state(self.random_state)
        self._root = _TreeNode(random_state=rng)

        thresholds = np.linspace(np.min(X, axis=0), np.max(X, axis=0), self.thresholds)

        distribution = np.zeros(self.n_classes_)
        for i in range(len(y)):
            distribution[y[i]] += 1

        entropy = _entropy(distribution, distribution.sum())

        self._root.build_tree(
            X,
            y,
            thresholds,
            entropy,
            distribution,
            0,
            self.max_depth,
            self.n_classes_,
            False,
        )

        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_instances, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        check_is_fitted(self)

        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        """Probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : 2d ndarray or DataFrame of shape = [n_instances, n_attributes]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        if isinstance(X, np.ndarray) and len(X.shape) == 3 and X.shape[1] == 1:
            X = np.reshape(X, (X.shape[0], -1))
        elif not isinstance(X, np.ndarray) or len(X.shape) > 2:
            raise ValueError(
                "ContinuousIntervalTree is not a time series classifier. "
                "A valid sklearn input such as a 2d numpy array is required."
                "Sparse input formats are currently not supported."
            )
        X = self._validate_data(X=X, reset=False, force_all_finite="allow-nan")

        dists = np.zeros((X.shape[0], self.n_classes_))
        for i in range(X.shape[0]):
            dists[i] = self._root.predict_proba(X[i], self.n_classes_)
        return dists

    def _predict_proba_cif(self, X, c22, intervals, dims, atts):
        """Embedded predict proba for the CIF classifier."""
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        n_instances, n_dims, series_length = X.shape

        dists = np.zeros((n_instances, self.n_classes_))
        for i in range(n_instances):
            dists[i] = self._root.predict_proba_cif(
                X[i].reshape((1, n_dims, series_length)),
                c22,
                intervals,
                dims,
                atts,
                self.n_classes_,
            )
        return dists

    def _predict_proba_drcif(
        self, X, X_p, X_d, c22, n_intervals, intervals, dims, atts
    ):
        """Embedded predict proba for the DrCIF classifier."""
        if not self._is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )
        n_instances, n_dims, series_length = X.shape

        dists = np.zeros((n_instances, self.n_classes_))
        for i in range(n_instances):
            r = [
                X[i].reshape((1, n_dims, series_length)),
                X_p[i].reshape((1, n_dims, X_p.shape[2])),
                X_d[i].reshape((1, n_dims, X_d.shape[2])),
            ]
            dists[i] = self._root.predict_proba_drcif(
                r,
                c22,
                n_intervals,
                intervals,
                dims,
                atts,
                self.n_classes_,
            )
        return dists

    def tree_node_splits_and_gain(self):
        """Recursively find the split and information gain for each tree node."""
        splits = []
        gains = []

        if self._root.best_split > -1:
            self._find_splits_gain(self._root, splits, gains)

        return splits, gains

    def _find_splits_gain(self, node, splits, gains):
        """Recursively find the split and information gain for each tree node."""
        splits.append(node.best_split)
        gains.append(node.best_gain)

        for next_node in node.children:
            if next_node.best_split > -1:
                self._find_splits_gain(next_node, splits, gains)


class _TreeNode:
    """ContinuousIntervalTree tree node."""

    def __init__(
        self,
        random_state=None,
    ):
        self.random_state = random_state

        self.best_split = -1
        self.best_threshold = 0
        self.best_gain = 0.000001
        self.best_margin = -1
        self.children = []
        self.leaf_distribution = []
        self.depth = -1

    def build_tree(
        self,
        X,
        y,
        thresholds,
        entropy,
        distribution,
        depth,
        max_depth,
        n_classes,
        leaf,
    ):
        from sktime.classification.sklearn._continuous_interval_tree_numba import (
            information_gain,
            margin_gain,
            remaining_classes,
            split_data,
        )

        self.depth = depth
        best_distributions = []
        best_entropies = []

        if leaf is False and remaining_classes(distribution) and depth < max_depth:
            for (_, att), threshold in np.ndenumerate(thresholds):
                (
                    info_gain,
                    distributions,
                    entropies,
                ) = information_gain(X, y, att, threshold, entropy, n_classes)

                if info_gain > self.best_gain:
                    self.best_split = att
                    self.best_threshold = threshold
                    self.best_gain = info_gain
                    self.best_margin = -1
                    best_distributions = distributions
                    best_entropies = entropies
                elif info_gain == self.best_gain and info_gain > 0.000001:
                    margin = margin_gain(X, att, threshold)
                    if self.best_margin == -1:
                        self.best_margin = margin_gain(
                            X, self.best_split, self.best_threshold
                        )

                    if margin > self.best_margin or (
                        margin == self.best_margin
                        and self.random_state.choice([True, False])
                    ):
                        self.best_split = att
                        self.best_threshold = threshold
                        self.best_margin = margin
                        best_distributions = distributions
                        best_entropies = entropies

        if self.best_split > -1:
            self.children = [None, None, None]

            left_idx, right_idx, missing_idx = split_data(
                X, self.best_split, self.best_threshold
            )

            if len(left_idx) > 0:
                self.children[0] = _TreeNode(random_state=self.random_state)
                self.children[0].build_tree(
                    X[left_idx],
                    y[left_idx],
                    thresholds,
                    best_entropies[0],
                    best_distributions[0],
                    depth + 1,
                    max_depth,
                    n_classes,
                    False,
                )
            else:
                self.children[0] = _TreeNode(random_state=self.random_state)
                self.children[0].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution,
                    depth + 1,
                    max_depth,
                    n_classes,
                    True,
                )

            if len(right_idx) > 0:
                self.children[1] = _TreeNode(random_state=self.random_state)
                self.children[1].build_tree(
                    X[right_idx],
                    y[right_idx],
                    thresholds,
                    best_entropies[1],
                    best_distributions[1],
                    depth + 1,
                    max_depth,
                    n_classes,
                    False,
                )
            else:
                self.children[1] = _TreeNode(random_state=self.random_state)
                self.children[1].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution,
                    depth + 1,
                    max_depth,
                    n_classes,
                    True,
                )

            if len(missing_idx) > 0:
                self.children[2] = _TreeNode(random_state=self.random_state)
                self.children[2].build_tree(
                    X[missing_idx],
                    y[missing_idx],
                    thresholds,
                    best_entropies[2],
                    best_distributions[2],
                    depth + 1,
                    max_depth,
                    n_classes,
                    False,
                )
            else:
                self.children[2] = _TreeNode(random_state=self.random_state)
                self.children[2].build_tree(
                    X,
                    y,
                    thresholds,
                    entropy,
                    distribution,
                    depth + 1,
                    max_depth,
                    n_classes,
                    True,
                )
        else:
            self.leaf_distribution = distribution / np.sum(distribution)

        return self

    def predict_proba(self, X, n_classes):
        if self.best_split > -1:
            if X[self.best_split] <= self.best_threshold:
                return self.children[0].predict_proba(X, n_classes)
            elif X[self.best_split] > self.best_threshold:
                return self.children[1].predict_proba(X, n_classes)
            else:
                return self.children[2].predict_proba(X, n_classes)
        else:
            return self.leaf_distribution

    def predict_proba_cif(self, X, c22, intervals, dims, atts, n_classes):
        from sktime.classification.sklearn._continuous_interval_tree_numba import (
            _drcif_feature,
        )

        if self.best_split > -1:
            interval = int(self.best_split / len(atts))
            att = self.best_split % len(atts)
            value = _drcif_feature(
                X, intervals[interval], dims[interval], atts[att], c22
            )
            value = value.round(8)
            value = np.nan_to_num(value, False, posinf=np.nan, neginf=np.nan)

            if value <= self.best_threshold:
                return self.children[0].predict_proba_cif(
                    X,
                    c22,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
            elif value > self.best_threshold:
                return self.children[1].predict_proba_cif(
                    X,
                    c22,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
            else:
                return self.children[2].predict_proba_cif(
                    X,
                    c22,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
        else:
            return self.leaf_distribution

    def predict_proba_drcif(
        self,
        X,
        c22,
        n_intervals,
        intervals,
        dims,
        atts,
        n_classes,
    ):
        from sktime.classification.sklearn._continuous_interval_tree_numba import (
            _drcif_feature,
        )

        if self.best_split > -1:
            rep = -1
            rep_sum = 0
            for i in range(len(X)):
                rep_sum += n_intervals[i] * len(atts)
                if self.best_split < rep_sum:
                    rep = i
                    break

            interval = int(self.best_split / len(atts))
            att = self.best_split % len(atts)

            value = _drcif_feature(
                X[rep], intervals[interval], dims[interval], atts[att], c22
            )
            value = value.round(8)
            value = np.nan_to_num(value, False, posinf=np.nan, neginf=np.nan)

            if value <= self.best_threshold:
                return self.children[0].predict_proba_drcif(
                    X,
                    c22,
                    n_intervals,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
            elif value > self.best_threshold:
                return self.children[1].predict_proba_drcif(
                    X,
                    c22,
                    n_intervals,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
            else:
                return self.children[2].predict_proba_drcif(
                    X,
                    c22,
                    n_intervals,
                    intervals,
                    dims,
                    atts,
                    n_classes,
                )
        else:
            return self.leaf_distribution
