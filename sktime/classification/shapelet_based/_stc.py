# -*- coding: utf-8 -*-
"""Shapelet Transform Classifier.

Shapelet transform classifier pipeline that simply performs a (configurable) shapelet
transform then builds (by default) a rotation forest classifier on the output.
"""

__author__ = ["TonyBagnall", "a-pasos-ruiz", "MatthewMiddlehurst"]
__all__ = ["ShapeletTransformClassifier"]

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import class_distribution

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.classification.shapelet_based.dev.factories.shapelet_factory import (
    ShapeletFactoryIndependent,
)
from sktime.classification.shapelet_based.dev.filters.random_filter import RandomFilter
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.transformations.panel.shapelets import ContractedShapeletTransform
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import check_X_y


class ShapeletTransformClassifier(BaseClassifier):
    """Shapelet Transform Classifier.

    TODO
    Basic implementation along the lines of [1,2]

    Parameters
    ----------
    n_shapelets : int, default=10000

    max_shapelets : int or None, default=None

    max_shapelet_length : int or None, default=None

    estimator : BaseEstimator or None, default=None

    transform_limit_in_minutes : int, default=60
        Time contract to limit transform time in minutes for the shapelet transform,
        overriding n_shapelets. A value of 0 means n_shapelets is used.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_estimators.
        Default of 0 means n_estimators is used.
    save_transformed_data : bool, default=False
        Save the data transformed in fit for use in _get_train_probs.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    n_instances : int
        The number of train cases.
    n_dims : int
        The number of dimensions per case.
    classes_ : list
        The classes labels.
    transformed_data : list of shape (n_estimators) of ndarray
        The transformed dataset for all classifiers. Only saved when
        save_transformed_data is true.

    See Also
    --------
    RotationForest

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

    TODO
    Examples
    --------
    >>> from sktime.classification.shapelet_based import ShapeletTransformClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = ShapeletTransformClassifier(n_estimators=10,
    >>> transform_limit_in_minutes=0.025)
    >>> clf.fit(X_train, y_train)
    ShapeletTransformClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "coerce-X-to-numpy": False,
        "coerce-X-to-pandas": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": False,
        "capability:train_estimate": True,
        "capability:contractable": True,
    }

    def __init__(
        self,
        n_shapelets=10000,
        max_shapelets=None,
        max_shapelet_length=None,
        estimator=None,
        transform_limit_in_minutes=60,
        time_limit_in_minutes=0,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.n_shapelets = n_shapelets
        self.max_shapelets = max_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.estimator = estimator

        self.transform_limit_in_minutes = transform_limit_in_minutes
        # TODO
        self.time_limit_in_minutes = time_limit_in_minutes
        self.save_transformed_data = save_transformed_data

        self.random_state = random_state
        self.n_jobs = n_jobs

        self.n_instances = 0
        self.n_dims = 0
        self.n_classes = 0
        self.classes_ = []
        self.transformed_data = []

        self._max_shapelets = max_shapelets
        self._max_shapelet_length = max_shapelet_length
        self._transformer = None
        self._estimator = estimator
        self._n_jobs = n_jobs

        super(ShapeletTransformClassifier, self).__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_instances, self.n_dims = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        if self.max_shapelet_length is None:
            self._max_shapelet_length = X.applymap(lambda x: len(x)).to_numpy().min()

        if self.max_shapelets is None:
            self._max_shapelets = (
                10 * self.n_instances if 10 * self.n_instances < 1000 else 1000
            )

        # TODO
        self._transformer = (
            RandomFilter(
                shapelet_factory=ShapeletFactoryIndependent(),
                max_shapelet_length=self._max_shapelet_length,
                num_shapelets=self._max_shapelets,
                num_iterations=self.n_shapelets,
                #    time_contract_in_mins=self.transform_contract_in_mins,
                #    verbose=False,
                random_state=self.random_state,
            )
            if self.n_dims > 1
            else ContractedShapeletTransform(
                time_contract_in_mins=self.transform_limit_in_minutes,
                verbose=False,
                random_state=self.random_state,
            )
        )

        self._estimator = _clone_estimator(
            RotationForest(save_transformed_data=self.save_transformed_data)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if callable(m):
            self._estimator.n_jobs = self._n_jobs

        if self.save_transformed_data:
            self.transformed_data = self._transformer.fit_transform(X, y)
            self._estimator.fit(self.transformed_data, y)
        else:
            self._estimator.fit(self._transformer.fit_transform(X, y), y)

    def _predict(self, X):
        return self._estimator.predict(self._transformer.transform(X))

    def _predict_proba(self, X):
        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists

    # TODO
    def _get_train_probs(self, X, y):
        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_pandas=True)

        n_instances, n_dims = X.shape

        if n_instances != self.n_instances or n_dims != self.n_dims:
            raise ValueError(
                "n_instances, n_dims, series_length mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        if isinstance(self.estimator, RotationForest):
            return self._estimator._get_train_probs(X, y)
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
                X=self.transformed_data,
                y=y,
                cv=cv_size,
                method="predict_proba",
                n_jobs=self._n_jobs,
            )
