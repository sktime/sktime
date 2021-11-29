# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and RidgeClassifierCV estimator.
"""

__author__ = ["MatthewMiddlehurst", "victordremov"]
__all__ = ["RocketClassifier"]

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from sktime.classification.base import BaseClassifier
from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)


class RocketClassifier(BaseClassifier):
    """Classifier wrapped for the Rocket transformer using RidgeClassifierCV.

    This classifier simply transforms the input data using the Rocket [1]_
    transformer and builds a RidgeClassifierCV estimator using the transformed data.

    Parameters
    ----------
    num_kernels : int, default=10,000
        The number of kernels the for Rocket transform.
    rocket_transform : str, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket","minirocket","multirocket"]
    max_dilations_per_kernel : int, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, default=4
        MultiRocket only. The number of features per kernel.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.

    See Also
    --------
    Rocket

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/shapelet_based/ROCKETClassifier.java>`_.

    References
    ----------
    .. [1] Dempster, Angus, François Petitjean, and Geoffrey I. Webb. "Rocket:
       exceptionally fast and accurate time series classification using random
       convolutional kernels." Data Mining and Knowledge Discovery 34.5 (2020)

    Examples
    --------
    >>> from sktime.classification.kernel_based import RocketClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = RocketClassifier(num_kernels=500)
    >>> clf.fit(X_train, y_train)
    RocketClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        num_kernels=10000,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        n_jobs=1,
        random_state=None,
    ):
        self.num_kernels = num_kernels
        self.rocket_transform = rocket_transform
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.n_features_per_kernel = n_features_per_kernel

        self.n_jobs = n_jobs
        self.random_state = random_state

        self._pipeline = None

        super(RocketClassifier, self).__init__()

    def _fit(self, X, y):
        """Build a pipeline containing the Rocket transformer and RidgeClassifierCV.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
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
        ending in "_" and sets is_fitted flag to True.
        """
        _, n_dims, _ = X.shape

        if self.rocket_transform == "rocket":
            rocket = Rocket(
                num_kernels=self.num_kernels,
                random_state=self.random_state,
                n_jobs=self._threads_to_use,
            )
        elif self.rocket_transform == "minirocket":
            if n_dims > 1:
                rocket = MiniRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    random_state=self.random_state,
                    n_jobs=self._threads_to_use,
                )
            else:
                rocket = MiniRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    random_state=self.random_state,
                    n_jobs=self._threads_to_use,
                )
        elif self.rocket_transform == "multirocket":
            if n_dims > 1:
                rocket = MultiRocketMultivariate(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    random_state=self.random_state,
                    n_jobs=self._threads_to_use,
                )
            else:
                rocket = MultiRocket(
                    num_kernels=self.num_kernels,
                    max_dilations_per_kernel=self.max_dilations_per_kernel,
                    n_features_per_kernel=self.n_features_per_kernel,
                    random_state=self.random_state,
                    n_jobs=self._threads_to_use,
                )
        else:
            raise ValueError(f"Invalid Rocket transformer: {self.rocket_transform}")

        self._pipeline = rocket_pipeline = make_pipeline(
            rocket,
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        )
        rocket_pipeline.fit(X, y)

        return self

    def _predict(self, X):
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        return self._pipeline.predict(X)

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        dists = np.zeros((X.shape[0], self.n_classes_))
        preds = self._pipeline.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1

        return dists
