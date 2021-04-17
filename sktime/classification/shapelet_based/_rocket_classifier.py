# -*- coding: utf-8 -*-
""" RandOm Convolutional KErnel Transform (ROCKET)
"""

__author__ = ["Matthew Middlehurst", "Oleksii Kachaiev"]
__all__ = ["ROCKETClassifier"]

import numpy as np
from joblib import delayed, Parallel
from sklearn.base import clone
from sklearn.ensemble._base import _set_random_states
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.rocket import Rocket
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import check_X
from sktime.utils.validation.panel import check_X_y

import warnings


class ROCKETClassifier(BaseClassifier):
    """
    Classifier wrapped for the ROCKET transformer using RidgeClassifierCV as the
    base classifier.
    Allows the creation of an ensemble of ROCKET classifiers to allow for
    generation of probabilities as the expense of scalability.

    Parameters
    ----------
    num_kernels             : int, number of kernels for ROCKET transform
    (default=10,000)
    n_estimators            : int, ensemble size, optional (default=None). When set
    to None (default) or 1, the classifier uses a single estimator rather than ensemble
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)
    n_jobs                  : int, the number of jobs to run in parallel for `fit`,
    optional (default=1)

    Attributes
    ----------
    estimators_             : array of individual classifiers
    weights                 : weight of each classifier in the ensemble
    weight_sum              : sum of all weights
    n_classes               : extracted from the data

    Notes
    -----
    @article{dempster_etal_2019,
      author  = {Dempster, Angus and Petitjean, Francois and Webb,
      Geoffrey I},
      title   = {ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels},
      year    = {2019},
      journal = {arXiv:1910.13051}
    }

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/ROCKETClassifier.java

    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
    }

    def __init__(
        self,
        num_kernels=10000,
        ensemble=None,
        ensemble_size=25,
        random_state=None,
        n_estimators=None,
        n_jobs=1,
    ):
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        # for compatibility only
        self.ensemble = ensemble
        self.ensemble_size = ensemble_size

        # for compatibility only
        if ensemble is not None and n_estimators is None:
            self.n_estimators = ensemble_size
            warnings.warn(
                "ensemble and ensemble_size params are deprecated and will be "
                "removed in future releases, use n_estimators instead",
                PendingDeprecationWarning,
            )

        self.estimators_ = []
        self.weights = []
        self.weight_sum = 0

        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}

        super(ROCKETClassifier, self).__init__()

    def fit(self, X, y):
        """
        Build a single or ensemble of pipelines containing the ROCKET transformer and
        RidgeClassifierCV classifier.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        n_jobs = check_n_jobs(self.n_jobs)

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary[class_val] = index

        if self.n_estimators is not None and self.n_estimators > 1:
            base_estimator = _make_estimator(self.num_kernels, self.random_state)
            self.estimators_ = Parallel(n_jobs=n_jobs)(
                delayed(_fit_estimator)(
                    _clone_estimator(base_estimator, self.random_state), X, y
                )
                for _ in range(self.n_estimators)
            )
            for rocket_pipeline in self.estimators_:
                weight = rocket_pipeline.steps[1][1].best_score_
                self.weights.append(weight)
                self.weight_sum += weight
        else:
            base_estimator = _make_estimator(self.num_kernels, self.random_state)
            self.estimators_ = [_fit_estimator(base_estimator, X, y)]

        self._is_fitted = True
        return self

    def predict(self, X):
        if self.n_estimators is not None:
            rng = check_random_state(self.random_state)
            return np.array(
                [
                    self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                    for prob in self.predict_proba(X)
                ]
            )
        else:
            self.check_is_fitted()
            return self.estimators_[0].predict(X)

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X)

        if self.n_estimators is not None:
            sums = np.zeros((X.shape[0], self.n_classes))

            for n, clf in enumerate(self.estimators_):
                preds = clf.predict(X)
                for i in range(0, X.shape[0]):
                    sums[i, self.class_dictionary[preds[i]]] += self.weights[n]

            dists = sums / (np.ones(self.n_classes) * self.weight_sum)
        else:
            dists = np.zeros((X.shape[0], self.n_classes))
            preds = self.estimators_[0].predict(X)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists

    # for compatibility
    @property
    def classifiers(self):
        warnings.warn(
            "classifiers attribute is deprecated and will be removed "
            "in future releases, use estimators_ instead",
            PendingDeprecationWarning,
        )
        return self.estimators_


def _fit_estimator(estimator, X, y):
    return estimator.fit(X, y)


def _make_estimator(num_kernels, random_state):
    return make_pipeline(
        Rocket(num_kernels=num_kernels, random_state=random_state),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
    )


def _clone_estimator(base_estimator, random_state=None):
    estimator = clone(base_estimator)

    if random_state is not None:
        _set_random_states(estimator, random_state)

    return estimator
