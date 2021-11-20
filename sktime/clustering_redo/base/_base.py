# -*- coding: utf-8 -*-
"""Base class for clustering."""

__author__ = ["chrisholder", "TonyBagnall"]
__all__ = ["BaseClusterer"]

from abc import ABC, abstractmethod

import numpy as np

from sktime.base import BaseEstimator
from sktime.clustering.base._types import TimeSeriesPanel
from sktime.utils.validation.panel import check_X


class BaseClusterer(BaseEstimator, ABC):
    """Abstract base class for time series clusterer."""

    _tags = {
        "coerce-X-to-numpy": True,
        "coerce-X-to-pandas": True,
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
        "capability:multithreading": False,
    }

    def __init__(self):
        self._threads_to_use = 1
        self._is_fitted = False
        super(BaseClusterer, self).__init__()

    def fit(self, X: TimeSeriesPanel, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances,n_dimensions,series_length)) or pd.DataFrame (where each column
            is a dimension, each cell is a pd.Series (any number of dimensions, equal or
            unequal length series)) or List[pd.Dataframe].
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        self:
            Fitted estimator.
        """
        X = check_X(
            X,
            coerce_to_numpy=self.get_tag("coerce-X-to-numpy"),
            coerce_to_pandas=self.get_tag("coerce-X-to-pandas"),
            enforce_univariate=not self.get_tag("capability:multivariate"),
        )

        multithread = self.get_tag("capability:multithreading")
        if multithread:
            try:
                self._threads_to_use = check_n_jobs(self.n_jobs)
            except NameError:
                raise AttributeError(
                    "self.n_jobs must be set if capability:multithreading is True"
                )

        self._fit(X, y)

        self._is_fitted = True

        return self

    def predict(self, X: TimeSeriesPanel, y=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances,n_dimensions,series_length)) or pd.DataFrame (where each column
            is a dimension, each cell is a pd.Series (any number of dimensions, equal or
            unequal length series)) or List[pd.Dataframe].
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        self.check_is_fitted()
        X = check_X(
            X,
            coerce_to_numpy=self.get_tag("coerce-X-to-numpy"),
            coerce_to_pandas=self.get_tag("coerce-X-to-pandas"),
            enforce_univariate=not self.get_tag("capability:multivariate"),
        )

        return self._predict(X)

    def fit_predict(self, X: TimeSeriesPanel, y=None):
        """Compute cluster centers and predict cluster index for each time series.

        Convenience method; equivalent of calling fit(X) followed by predict(X)

        Parameters
        ----------
        X : np.ndarray (2d or 3d array of shape (n_instances, series_length) or shape
            (n_instances,n_dimensions,series_length)) or pd.DataFrame (where each column
            is a dimension, each cell is a pd.Series (any number of dimensions, equal or
            unequal length series)) or List[pd.Dataframe].
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """
        self.fit(X)
        return self.predict(X)

    @abstractmethod
    def _predict(self, X: np.ndarray, y=None) -> np.ndarray:
        ...

    @abstractmethod
    def _fit(self, X: np.ndarray, y=None) -> np.ndarray:
        ...
