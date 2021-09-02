# -*- coding: utf-8 -*-

__author__ = ["Markus Löning", "Satya Pattnaik"]

import numpy as np

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class ConditionalImputer(_SeriesToSeriesTransformer):

    """Conditonal Imputer that uses a Sktime Imputer to
       impute values set to nan by the detector.
    Parameters
    ----------
    imputer : Sktime Imputer
    detector : Sktime Annotator/Detector
    """

    _tags = {"skip-inverse-transform": True}

    def __init__(self, imputer, detector):
        self.imputer = imputer
        self.detector = detector

    def fit(self, Z, X=None):
        """
        Fit to training data.
        Parameters
        ----------
        Z : pd.Series
            Target time series to which to fit the ConditionalImputer.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """

        z = check_series(Z, enforce_univariate=True)

        self.detector_ = self.detector.fit(z)

        z = z.to_numpy().reshape(-1, 1)
        self.imputer_ = self.imputer.fit(z)

        return self

    def transform(self, Z, X=None):
        """
        Returns a transformed version of Z.
        Parameters
        ----------
        Z : pd.Series
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        Zt : pd.Series
            Transformed(Conditionally Imputed) time series.
        """

        z = check_series(Z, enforce_univariate=True)

        outliers = self.detector_.transform(z)

        Zt = Z.copy()
        Zt.iloc[outliers[outliers].index] = np.nan

        z = Zt
        zt = self.imputer_.transform(z)
        return zt

    def fit_transform(self, Z, X=None):
        """
        Fit to training data.
        Returns a transformed version of Z.
        Parameters
        ----------
        Z : pd.Series
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        Zt : pd.Series
            Transformed(Conditionally Imputed) time series.
        """

        z = check_series(Z, enforce_univariate=True)

        self.detector_ = self.detector.fit(z)

        outliers = self.detector_.transform(z)

        Zt = Z.copy()
        Zt[outliers[outliers].index] = np.nan

        z = Zt
        self.imputer_ = self.imputer.fit(z)
        zt = self.imputer_.fit_transform(z)
        return zt
