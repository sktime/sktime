#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements base class for time series forests."""

__author__ = ["mloning", "AyushmaanSeth"]
__all__ = ["BaseTimeSeriesForest"]

from abc import abstractmethod
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from numpy import float64 as DOUBLE
from sklearn.base import clone
from sklearn.ensemble._forest import (
    MAX_INT,
    BaseForest,
    _generate_sample_indices,
    _get_n_samples_bootstrap,
)
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_array, check_random_state, compute_sample_weight

from sktime.utils.random_state import set_random_state
from sktime.utils.warnings import warn


def _parallel_build_trees(
    tree,
    forest,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
):
    """Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))  # noqa: T201

    # name of step of final estimator in pipeline
    final_estimator = tree.steps[-1][1]

    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(
            final_estimator.random_state, n_samples, n_samples_bootstrap
        )
        sample_counts = np.bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == "subsample":
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                curr_sample_weight *= compute_sample_weight("auto", y, indices)
        elif class_weight == "balanced_subsample":
            curr_sample_weight *= compute_sample_weight("balanced", y, indices)
        tree.fit(X, y)
    else:
        tree.fit(X, y)

    return tree


class BaseTimeSeriesForest(BaseForest):
    """Base class for forests of trees."""

    @abstractmethod
    def __init__(
        self,
        base_estimator,
        n_estimators=100,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        super().__init__(
            base_estimator, n_estimators=n_estimators, estimator_params=estimator_params
        )
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `_estimator` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self._estimator)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})

        if random_state is not None:
            set_random_state(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def _fit(self, X, y, sample_weight=None):
        """Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """
        from joblib import Parallel, delayed
        from scipy.sparse import issparse

        # Validate or convert input data
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_columns = X.shape[1]
        self.n_features = X.shape[1] if X.ndim == 2 else 1

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
                obj=self,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0], max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees.",
                obj=self,
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: for standard random forests, the threading
            # backend is preferred as the Cython code for fitting the trees
            # is internally releasing the Python GIL making threading more
            # efficient than multiprocessing in that case.
            # However, in this case,for fitting pipelines in parallel,
            # multiprocessing is more efficient.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_parallel_build_trees)(
                    t,
                    self,
                    X,
                    y,
                    sample_weight,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        self._is_fitted = True
        return self

    def apply(self, X):
        """Abstract method that is implemented by concrete estimators."""
        raise NotImplementedError()

    def decision_path(self, X):
        """Decision path of decision tree.

        Abstract method that is implemented by concrete estimators.
        """
        raise NotImplementedError()

    def _validate_X_predict(self, X):
        n_features = X.shape[1] if X.ndim == 2 else 1
        if self.n_columns != n_features:
            raise ValueError(
                "Number of features of the model must "
                "match the input. Model n_features is %s and "
                "input n_features is %s " % (self.n_columns, n_features)
            )

        return X

    @property
    def feature_importances_(self, normalise_time_points=False):
        """Compute feature importances for time series forest."""
        # assumes particular structure of clf,
        # with each tree consisting of a particular pipeline,
        # as in modular tsf
        from sktime.transformations.panel.summarize import (
            RandomIntervalFeatureExtractor,
        )

        if not isinstance(
            self.estimators_[0].steps[0][1], RandomIntervalFeatureExtractor
        ):
            raise NotImplementedError(
                "RandomIntervalFeatureExtractor must"
                " be used as the transformer,"
                " which must be the first step"
                " in the base estimator."
            )

        # get series length, assuming same length series
        tree = self.estimators_[0]
        transformer = tree.steps[0][1]
        time_index = transformer._time_index
        n_timepoints = len(time_index)

        # get feature names, features are the same for all trees
        feature_names = [feature.__name__ for feature in transformer.features]
        n_features = len(feature_names)

        # get intervals from transformer,
        # the number of intervals is the same for all trees

        intervals = transformer.intervals_
        n_intervals = len(intervals)

        # get number of estimators
        n_estimators = len(self.estimators_)

        # preallocate array for feature importances
        fis = np.zeros((n_timepoints, n_features))
        if normalise_time_points:
            fis_count = np.zeros((n_timepoints, n_features))

        for i in range(n_estimators):
            # select tree
            tree = self.estimators_[i]
            transformer = tree.steps[0][1]
            classifier = tree.steps[-1][1]

            # get intervals from transformer
            intervals = transformer.intervals_

            # get feature importances from classifier
            fi = classifier.feature_importances_

            for k in range(n_features):
                for j in range(n_intervals):
                    # get start and end point from interval
                    start, end = intervals[j]

                    # get time index for interval
                    interval_time_points = np.arange(start, end)

                    # get index for feature importances,
                    # assuming particular order of features

                    column_index = (k * n_intervals) + j

                    # add feature importance for all time points of interval
                    fis[interval_time_points, k] += fi[column_index]
                    if normalise_time_points:
                        fis_count[interval_time_points, k] += 1

        # normalise by number of estimators and number of intervals
        fis = fis / n_estimators / n_intervals

        # format output
        fis = pd.DataFrame(fis, columns=feature_names, index=time_index)

        if normalise_time_points:
            fis_count = fis_count / n_estimators / n_intervals
            fis_count = pd.DataFrame(fis_count, columns=feature_names, index=time_index)
            fis /= fis_count

        return fis
