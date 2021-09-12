# -*- coding: utf-8 -*-


import numpy as np

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sklearn import clone

__author__ = ["mloning", "satya-pattnaik"]
_required_parameters = ["imputer", "annotator"]


class ConditionalImputer(_SeriesToSeriesTransformer):

    """Conditonal Imputer that uses a Sktime Imputer to
       impute values set to nan by the annotator.
    Parameters
    ----------
    imputer : Sktime Imputer
    annotator : Sktime Annotator/annotator
    """

    _tags = {"skip-inverse-transform": True}

    def __init__(self, imputer, annotator):
        self.imputer = imputer
        self.annotator = annotator

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

        self.annotator_ = clone(self.annotator)
        self.annotator_.set_params({"fmt": "dense"})
        self.annotator_.fit(z, X)

        self.imputer_ = clone(self.imputer)
        self.imputer_.fit(z, X)

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

        outliers = self.annotator_.predict(z, X)

        z.iloc[outliers] = np.nan
        z = self.imputer_.transform(z, X)
        return z
