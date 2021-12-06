# -*- coding: utf-8 -*-
"""TSFresh Classifier.

Pipeline classifier using the TSFresh transformer and an estimator.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["FreshPRINCE"]

import numpy as np
from sklearn.pipeline import Pipeline

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.classification.feature_based._tsfresh_classifier import TSFreshClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor


class FreshPRINCE(BaseClassifier):
    """Time Series Feature Extraction based on Scalable Hypothesis Tests classifier.

    This classifier simply transforms the input data using the TSFresh [1]
    transformer and builds a provided estimator using the transformed data.

    Parameters
    ----------
    default_fc_parameters : str, default="comprehensive"
        Set of TSFresh features to be extracted, options are "minimal", "efficient" or
        "comprehensive".
    n_estimators : int, default=200

    verbose : int, default=0
        level of output printed to the console (for information only)
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    chunksize : int or None, default=None
        Number of series processed in each parallel TSFresh job, should be optimised
        for efficient parallelisation.
    random_state : int or None, default=None
        Seed for random, integer.

    Attributes
    ----------
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.

    See Also
    --------
    TSFreshFeatureExtractor, TSFreshClassifier, RotationForest

    References
    ----------
    .. [1] Christ, Maximilian, et al. "Time series feature extraction on basis of
        scalable hypothesis tests (tsfresh–a python package)." Neurocomputing 307
        (2018): 72-77.
        https://www.sciencedirect.com/science/article/pii/S0925231218304843

    Examples
    --------
    >>> from sktime.classification.feature_based import TSFreshClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = TSFreshClassifier(
    ...     default_fc_parameters="minimal",
    ...     estimator=RandomForestClassifier(n_estimators=10),
    ... )
    >>> clf.fit(X_train, y_train)
    TSFreshClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "capability:train_estimate": True,
    }

    def __init__(
        self,
        default_fc_parameters="comprehensive",
        n_estimators=200,
        verbose=0,
        n_jobs=1,
        chunksize=None,
        random_state=None,
    ):
        self.default_fc_parameters = default_fc_parameters
        self.n_estimators = n_estimators

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

        self.n_instances = 0
        self.n_dims = 0
        self._estimator = None

        super(FreshPRINCE, self).__init__()

    def _fit(self, X, y):
        """Fit a pipeline on cases (X,y), where y is the target variable.

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
        self._estimator = TSFreshClassifier(
            default_fc_parameters=self.default_fc_parameters,
            estimator=RotationForest(n_estimators=self.n_estimators),
            n_jobs=self._threads_to_use,
            chunksize=self.chunksize,
            verbose=self.verbose,
        )

        self._estimator.fit(X, y)

        return self

    def _predict(self, X):
        """Predict class values of n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        return self._estimator.predict(X)

    def _predict_proba(self, X):
        """Predict class probabilities for n instances in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        return self._estimator.predict_proba(X)

    def _get_train_probs(self, X, y):
        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_pandas=True)

        n_instances, n_dims = X.shape

        if n_instances != self.n_instances_ or n_dims != self.n_dims_:
            raise ValueError(
                "n_instances, n_dims, series_length mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        if isinstance(self.estimator, RotationForest) or self.estimator is None:
            return self._estimator._get_train_probs(self.transformed_data_, y)
        else:
            m = getattr(self._estimator, "predict_proba", None)
            if not callable(m):
                raise ValueError("Estimator must have a predict_proba method.")

            cv_size = 10
            _, counts = np.unique(y, return_counts=True)
            min_class = np.min(counts)
            if min_class < cv_size:
                cv_size = min_class

            estimator = _clone_estimator(self.estimator, self.random_state)

            return cross_val_predict(
                estimator,
                X=self.transformed_data_,
                y=y,
                cv=cv_size,
                method="predict_proba",
                n_jobs=self._threads_to_use,
            )
