#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from sktime.base import BaseEstimator


class BaseSeriesToSeriesTransformer(BaseEstimator):
    """Base class for series-to-series tranformers"""

    def __init__(self):
        super(BaseSeriesToSeriesTransformer, self).__init__()

    def fit(self, y_train, **fit_params):
        """Fit to data.

        Parameters
        ----------
        y_train : pd.Series
        fit_params : dict

        Returns
        -------
        self : an instance of self
        """
        self._is_fitted = True
        return self

    def fit_transform(self, y_train, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to y_train returns a transformed y_train.

        Parameters
        ----------
        y_train : pd.Series

        Returns
        -------
        yt : pd.Series
            Transformed time series.
        """
        return self.fit(y_train, **fit_params).transform(y_train)

    def transform(self, y, **transform_params):
        """Transform data.
        Returns a transformed version of y.

        Parameters
        ----------
        y : pd.Series

        Returns
        -------
        yt : pd.Series
            Transformed time series.
        """
        raise NotImplementedError("abstract method")

    def inverse_transform(self, y, **transform_params):
        """Inverse transform data.
        Returns a transformed version of y.

        Parameters
        ----------
        y : pd.Series

        Returns
        -------
        yt : pd.Series
            Inverse-transformed time series.
        """
        raise NotImplementedError("abstract method")

    def update(self, y_new, update_params=False):
        """Update fitted parameters

         Parameters
         ----------
         y_new : pd.Series
         update_params : bool, optional (default=False)

         Returns
         -------
         self : an instance of self
         """
        raise NotImplementedError("abstract method")
