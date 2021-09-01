# -*- coding: utf-8 -*-
"""Shapelet Transform Classifier.

Wrapper implementation of a shapelet transform classifier pipeline that simply
performs a (configurable) shapelet transform then builds (by default) a random
forest.
"""

__author__ = ["TonyBagnall", "a-pasos-ruiz", "MatthewMiddlehurst"]
__all__ = ["ShapeletTransformClassifier"]

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import class_distribution

from sktime.base._base import _clone_estimator
from sktime.classification.base import BaseClassifier
from sktime.classification.shapelet_based.dev.factories.shapelet_factory import (
    ShapeletFactoryIndependent,
)
from sktime.classification.shapelet_based.dev.filters.random_filter import RandomFilter
from sktime.transformations.panel.shapelets import ContractedShapeletTransform
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import check_X_y


class ShapeletTransformClassifier(BaseClassifier):
    """Shapelet Transform Classifier.

    TODO
    Basic implementation along the lines of [1,2]

    Parameters
    ----------
    transform_contract_in_mins : int, search time for shapelets, optional
    (default = 60)
    n_estimators               :       200,
    random_state               :  int, seed for random, optional (default = none)

    Attributes
    ----------
    TO DO

    Notes
    -----
    ..[1] Jon Hills et al., "Classification of time series by
    shapelet transformation",
        Data Mining and Knowledge Discovery, 28(4), 851--881, 2014
    https://link.springer.com/article/10.1007/s10618-013-0322-1
    ..[2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform
    for Multiclass Time Series Classification",
    Transactions on Large-Scale Data and Knowledge Centered
      Systems, 32, 2017
    https://link.springer.com/chapter/10.1007/978-3-319-22729-0_20
    Java Version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/classifiers/shapelet_based/ShapeletTransformClassifier.java
    """

    _tags = {
        # "coerce-X-to-numpy": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": True,
        "contractable": True,
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
        self.time_limit_in_minutes = time_limit_in_minutes
        self.save_transformed_data = save_transformed_data

        self.random_state = random_state
        self.n_jobs = n_jobs

        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.n_classes_ = 0
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

        self.n_instances, self.n_dims, self.series_length = X.shape
        self.n_classes_ = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        if self.max_shapelet_length is None:
            self._max_shapelet_length = self.series_length

        if self.max_shapelets is None:
            self._max_shapelets = 10 * len(X) if 10 * len(X) < 1000 else 1000

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
            RandomForestClassifier(n_estimators=200, n_jobs=self._n_jobs)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

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
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, np.where(self.classes_ == preds[i])] = 1
            return dists

    def _get_train_probs(self, X, y):
        self.check_is_fitted()
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        n_instances, n_dims, series_length = X.shape

        if (
            n_instances != self.n_instances
            or n_dims != self.n_dims
            or series_length != self.series_length
        ):
            raise ValueError(
                "n_instances, n_dims, series_length mismatch. X should be "
                "the same as the training data used in fit for generating train "
                "probabilities."
            )

        if not self.save_transformed_data:
            raise ValueError("Currently only works with saved transform data from fit.")

        m = getattr(self._estimator, "predict_proba", None)
        if not callable(m):
            raise ValueError("Estimator must have a predict_proba method.")

        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class

        classifier = _clone_estimator(
            RandomForestClassifier(n_estimators=200)
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        return cross_val_predict(
            classifier, X=self.transformed_data, y=y, cv=cv_size, method="predict_proba"
        )
