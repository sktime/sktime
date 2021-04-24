# -*- coding: utf-8 -*-
import warnings
import numpy as np
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series

__author__ = ["Markus Löning", "Satya Pattnaik"]


class PyODOutlierDetector(_SeriesToSeriesTransformer):
    """Transformer that applies Outlier Detection to a
    univariate time series.

    Provides a simple wrapper around ``PyOD``.

    Parameters
    ----------
    estimator : PyOD estimator
        See ``https://pyod.readthedocs.io/en/latest/`` documentation for a detailed
        description of all options.
    """

    _tags = {
        "univariate-only": True,
        "fit-in-transform": True,
        "handles-missing-data": True,
    }

    def __init__(self, estimator):
        self.estimator = estimator  # pyod estimator

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
        z = check_series(Z, enforce_univariate=True)
        z = z.to_numpy().reshape(-1, 1)
        self.estimator_ = self.estimator.fit(z)
        return self

    def transform(self, Z, X=None):
        """Returns a transformed version of Z.

        Parameters
        ----------
        Z : pd.Series
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        Zt : pd.Series
            Transformed(by PyOD Estimator) time series.
        """

        z = check_series(Z, enforce_univariate=True)

        if z.isnull().values.any():
            warnings.warn(
                """Series contains nan values, more nan might be
                added if there are outliers"""
            )

        z = z.to_numpy().reshape(-1, 1)
        outliers = self.estimator_.predict(z)

        outliers = np.where(outliers)
        Zt = Z.copy()
        Zt.iloc[:] = False
        Zt.iloc[outliers] = True
        return Zt
