# -*- coding: utf-8 -*-
"""
Dimension selection pipeline.

Assign a dimension selection method and any multivariate time series classifier
"""

__author__ = ["AlejandroPasosRuiz"]
__all__ = ["DSPipeline"]

from sklearn.pipeline import make_pipeline

from sktime.classification.base import BaseClassifier


class DSPipeline(BaseClassifier):
    """Dimension selection pipeline."""

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
    }

    def __init__(
        self,
        random_state=None,
        n_jobs=1,
        time_limit_in_minutes=0,
        ds_train_time=0,
        ds_num_selected_dimensions=0,
        ds_num_dimensions=0,
        ds_transformer=None,
        ds_classifier=None,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.time_limit_in_minutes = time_limit_in_minutes
        self.ds_train_time = ds_train_time
        self.ds_num_selected_dimensions = ds_num_selected_dimensions
        self.ds_num_dimensions = ds_num_dimensions
        self.ds_transformer = ds_transformer
        self.ds_classifier = ds_classifier
        self._pipeline = None
        super(DSPipeline, self).__init__()

    def _fit(self, X, y):
        _, n_dims, _ = X.shape
        self._pipeline = make_pipeline(self.ds_transformer, self.ds_classifier)
        self._pipeline.fit(X, y)
        self.ds_num_dimensions = n_dims
        self.ds_train_time = self.ds_transformer.train_time
        self.ds_num_selected_dimensions = len(self.ds_transformer.dimensions_selected)
        return self

    def _predict(self, X):
        return self._pipeline.predict(X)
