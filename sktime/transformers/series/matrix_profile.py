# -*- coding: utf-8 -*-

__all__ = ["MatrixProfileTransformer"]

import pandas as pd
from sktime.transformers.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sktime.utils.check_imports import _check_soft_dependencies

_check_soft_dependencies("stumpy")

import stumpy  # noqa: E402


def _stumpy_series(time_series, window_size):
    matrix_profile = stumpy.stump(time_series, m=window_size)
    return matrix_profile


class MatrixProfileTransformer(_SeriesToSeriesTransformer):
    def __init__(self, m=10):
        self.m = m  # subsequence length
        super(MatrixProfileTransformer, self).__init__()

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """
        Takes as input a time series dataset and returns the matrix profile
        for each single time series of the dataset.

        Parameters
        ----------
        X: pandas.DataFrame
           Time series dataset.

        Returns
        -------
        Xt: pandas.DataFrame
            Dataframe with the same number of rows as the input.
            The number of columns equals the number of subsequences
            of the desired length in each time series.
        """
        self.check_is_fitted()
        X = check_series(X, enforce_univariate=True)
        n_instances = X.shape[0]
        Xt = pd.DataFrame([_stumpy_series(X[i], self.m) for i in range(n_instances)])
        return Xt


df = pd.read_csv("/home/utsav/Downloads/timeseries.csv")
matrixProfileTransformer = MatrixProfileTransformer(3)
matrixProfileTransformer.transform(df)
# print(df)
