# -*- coding: utf-8 -*-
"""Arsenal, an ensemble of ROCKET classifiers."""

__author__ = ["Matthew Middlehurst", "Oleksii Kachaiev"]
__all__ = ["Arsenal"]

import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
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


class Arsenal(BaseClassifier):
    """Ensemble of ROCKET transformers using RidgeClassifierCV base classifier.

    Allows for generation of probabilities at the expense of scalability.

    Parameters
    ----------
    num_kernels             : int, number of kernels for ROCKET transform
    (default=2,000)
    n_estimators            : int, ensemble size, optional (default=25)
    n_jobs                  : int, the number of jobs to run in parallel for both `fit`
    and `predict`. ``-1`` means using all processors
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

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
    tsml/classifiers/shapelet_based/Arsenal.java

    """

    # Capability tags
    capabilities = {
        "multivariate": True,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        num_kernels=2000,
        n_estimators=25,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimators_ = []
        self.weights = []
        self.weight_sum = 0

        self.n_classes = 0
        self.classes_ = []
        self.class_dictionary = {}

        super(Arsenal, self).__init__()

    def fit(self, X, y):
        """Build an ensemble ROCKET transformer and RidgeClassifierCV classifier.

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
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary[classVal] = index

        base_estimator = _make_estimator(self.num_kernels, self.random_state)
        self.estimators_ = Parallel(n_jobs=n_jobs)(
            delayed(_fit_estimator)(
                _clone_estimator(base_estimator, self.random_state, i), X, y
            )
            for i in range(self.n_estimators)
        )

        self.weights = []
        self.weight_sum = 0
        for rocket_pipeline in self.estimators_:
            weight = rocket_pipeline.steps[1][1].best_score_
            self.weights.append(weight)
            self.weight_sum += weight

        self._is_fitted = True
        return self

    def predict(self, X):
        """Find predictions for all cases in X. Built on top of predict_proba.

        Parameters
        ----------
        X : The training input samples. array-like or pandas data frame.
        If a Pandas data frame is passed, a check is performed that it only
        has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances]
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : The training input samples. array-like or sparse matrix of shape
        = [n_test_instances, series_length]
            If a Pandas data frame is passed (sktime format) a check is
            performed that it only has one column.
            If not, an exception is thrown, since this classifier does not
            yet have
            multivariate capability.

        Returns
        -------
        output : array of shape = [n_test_instances, num_classes] of
        probabilities
        """
        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True)

        sums = np.zeros((X.shape[0], self.n_classes))

        for n, clf in enumerate(self.estimators_):
            preds = clf.predict(X)
            for i in range(0, X.shape[0]):
                sums[i, self.class_dictionary[preds[i]]] += self.weights[n]

        return np.around(sums / (np.ones(self.n_classes) * self.weight_sum), 8)

    def _get_train_probs(self, X):
        return 0


def _fit_estimator(estimator, X, y):
    return estimator.fit(X, y)


def _make_estimator(num_kernels, random_state):
    return make_pipeline(
        Rocket(num_kernels=num_kernels, random_state=random_state),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
    )


def _clone_estimator(base_estimator, random_state=None, idx=0):
    estimator = clone(base_estimator)

    if random_state is not None:
        random_state = 255 if random_state == 0 else random_state
        _set_random_states(estimator, random_state * 37 * (idx + 1))

    return estimator
