# -*- coding: utf-8 -*-
"""Shapelet Transform Classifier.

Shapelet transform classifier pipeline that simply performs a (configurable) shapelet
transform then builds (by default) a rotation forest classifier on the output.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["ShapeletTransformClassifier"]

import numpy as np
from sklearn.model_selection import cross_val_predict

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.utils.validation.panel import check_X_y


class ShapeletTransformClassifier(BaseClassifier):
    """Shapelet Transform Classifier.

    Implementation of the binary shapelet transform classifier along the lines
    of [1]_[2]_. Transforms the data using the configurable shapelet transform and then
    builds a rotation forest classifier.
    As some implementations and applications contract the classifier solely, contracting
    is available for the transform only and both classifier and transform.

    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        The number of candidate shapelets to be considered for the final transform.
        Filtered down to <= max_shapelets, keeping the shapelets with the most
        information gain.
    max_shapelets : int or None, default=None
        Max number of shapelets to keep for the final transform. Each class value will
        have its own max, set to n_classes / max_shapelets. If None uses the min between
        10 * n_instances and 1000
    max_shapelet_length : int or None, default=None
        Lower bound on candidate shapelet lengths for the transform.
    estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble, can be supplied a sklearn BaseEstimator. If
        None a default RotationForest classifier is used.
    transform_limit_in_minutes : int, default=0
        Time contract to limit transform time in minutes for the shapelet transform,
        overriding n_shapelets. A value of 0 means n_shapelets is used.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_shapelet_samples and
        transform_limit_in_minutes. The estimator will only be contracted if a
        time_limit_in_minutes parameter is present. Default of 0 means n_estimators or
        transform_limit_in_minutes is used.
    contract_max_n_shapelet_samples : int, default=np.inf
        Max number of shapelets to extract when contracting the transform with
        transform_limit_in_minutes or time_limit_in_minutes.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    batch_size : int or None, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets in the transform.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    transformed_data_ : list of shape (n_estimators) of ndarray
        The transformed dataset for all classifiers. Only saved when
        save_transformed_data is true.

    See Also
    --------
    RandomShapeletTransform

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/classifiers/shapelet_based/ShapeletTransformClassifier.java>`_.

    References
    ----------
    .. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
       Data Mining and Knowledge Discovery, 28(4), 851--881, 2014.
    .. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
       Series Classification", Transactions on Large-Scale Data and Knowledge Centered
       Systems, 32, 2017.

    Examples
    --------
    >>> from sktime.classification.shapelet_based import ShapeletTransformClassifier
    >>> from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = ShapeletTransformClassifier(
    ...     estimator=RotationForest(n_estimators=3),
    ...     n_shapelet_samples=500,
    ...     max_shapelets=20,
    ...     batch_size=100,
    ... )
    >>> clf.fit(X_train, y_train)
    ShapeletTransformClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        n_shapelet_samples=10000,
        max_shapelets=None,
        max_shapelet_length=None,
        estimator=None,
        transform_limit_in_minutes=0,
        time_limit_in_minutes=0,
        contract_max_n_shapelet_samples=np.inf,
        save_transformed_data=False,
        n_jobs=1,
        batch_size=100,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.estimator = estimator

        self.transform_limit_in_minutes = transform_limit_in_minutes
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples
        self.save_transformed_data = save_transformed_data

        self.random_state = random_state
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        self.n_instances_ = 0
        self.n_dims_ = 0
        self.series_length_ = 0
        self.transformed_data_ = []

        self._transformer = None
        self._estimator = estimator
        self._transform_limit_in_minutes = 0
        self._classifier_limit_in_minutes = 0

        super(ShapeletTransformClassifier, self).__init__()

    def _fit(self, X, y):
        """Fit STC to training data.

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
        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.time_limit_in_minutes > 0:
            # contracting 2/3 transform (with 1/5 of that taken away for final
            # transform), 1/3 classifier
            third = self.time_limit_in_minutes / 3
            self._classifier_limit_in_minutes = third
            self._transform_limit_in_minutes = (third * 2) / 5 * 4
        elif self.transform_limit_in_minutes > 0:
            self._transform_limit_in_minutes = self.transform_limit_in_minutes

        self._transformer = RandomShapeletTransform(
            n_shapelet_samples=self.n_shapelet_samples,
            max_shapelets=self.max_shapelets,
            max_shapelet_length=self.max_shapelet_length,
            time_limit_in_minutes=self._transform_limit_in_minutes,
            contract_max_n_shapelet_samples=self.contract_max_n_shapelet_samples,
            n_jobs=self.n_jobs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )

        self._estimator = _clone_estimator(
            RotationForest() if self.estimator is None else self.estimator,
            self.random_state,
        )

        if isinstance(self._estimator, RotationForest):
            self._estimator.save_transformed_data = self.save_transformed_data

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._threads_to_use

        m = getattr(self._estimator, "time_limit_in_minutes", None)
        if m is not None and self.time_limit_in_minutes > 0:
            self._estimator.time_limit_in_minutes = self._classifier_limit_in_minutes

        X_t = self._transformer.fit_transform(X, y).to_numpy()

        if self.save_transformed_data:
            self.transformed_data_ = X_t

        self._estimator.fit(X_t, y)

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
        X_t = self._transformer.transform(X).to_numpy()

        return self._estimator.predict(X_t)

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
        X_t = self._transformer.transform(X).to_numpy()

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_t)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(X_t)
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists

    def _get_train_probs(self, X, y):
        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_pandas=True)

        n_instances, n_dims = X.shape

        if n_instances != self.n_instances_ or n_dims != self.n_dims_:
            raise ValueError(
                "n_instances, n_dims mismatch. X should be "
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
