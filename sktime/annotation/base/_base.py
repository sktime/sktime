#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)


from sktime.base import BaseEstimator


class BaseAnnotator(BaseEstimator):
    """Base annotator

    The base annotator specifies the methods and method
    signatures that all annotators have to implement.

    Specific implementations of these methods is deferred to concrete
    annotators.
    """

    def __init__(self):
        self._is_fitted = False
        super(BaseAnnotator, self).__init__()

    def fit(self, Z, X=None):
        """Fit to training data.

        Parameters
        ----------
        Z : pd.Series
            Target time series to which to fit the annotator.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

    def predict(self, fh=None, X=None):
        """Returns a transformed version of Z.

        Parameters
        ----------
        Z : pd.Series
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        z : pd.Series
            Transformed(by annotator) time series.
        """
        raise NotImplementedError("abstract method")
