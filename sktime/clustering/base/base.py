# -*- coding: utf-8 -*-
"""Base classes for clustering."""
__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = [
    "BaseClusterer",
    "BaseClusterCenterInitializer",
    "BaseClusterAverage",
    "BaseApproximate",
    "DataFrame",
    "NumpyArray",
    "NumpyOrDF",
    "NumpyRandomState",
    "CenterCalculatorFunc",
]

import pandas as pd
import numpy as np
from sktime.base import BaseEstimator
from sktime.utils.data_processing import from_nested_to_2d_array
from typing import Union, Callable

DataFrame = pd.DataFrame
NumpyArray = np.ndarray
NumpyOrDF = Union[DataFrame, NumpyArray]
NumpyRandomState = Union[np.random.RandomState, int]
CenterCalculatorFunc = Callable[[NumpyArray], NumpyArray]


class BaseClusterer(BaseEstimator):
    """Base Clusterer"""

    def __init__(self):
        super(BaseClusterer, self).__init__()

    def fit(self, X: NumpyOrDF, y=None):
        """
        Fit the clustering algorithm on the dataset X

        Parameters
        ----------
        X: 2D np.array with shape (n_instances, n_timepoints)
           or pd.DataFrame in nested format
            panel of univariate time series to train the clustering model on

        y: ignored, exists for API consistency reasons

        Returns
        -------
        reference to self
        """
        if isinstance(X, pd.DataFrame):
            X = from_nested_to_2d_array(X, return_numpy=True)

        self._check_params(X)
        self._fit(X)

        self._is_fitted = True
        return self

    def predict(self, X: NumpyOrDF, y=None) -> NumpyArray:
        """
        Return cluster center index for data samples.

        Parameters
        ----------
        X: 2D np.array with shape (n_instances, n_timepoints)
           or pd.DataFrame in nested format
            panel of time series to cluster

        y: ignored, exists for API consistency reasons

        Returns
        -------
        Numpy_Array: 1D np.array of length n_instances
            Index of the cluster each sample belongs to
        """
        self.check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = from_nested_to_2d_array(X, return_numpy=True)

        self._check_params(X)
        return self._predict(X)

    def _fit(self, X: NumpyArray):
        """
        Fit the clustering algorithm on the dataset X

            core logic

        Parameters
        ----------
        X: 2D np.array with shape (n_instances, n_timepoints)
            panel of univariate time series to train the clustering model on

        Returns
        -------
        reference to self
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X: NumpyArray) -> NumpyArray:
        """
        Return cluster center index for data samples.

            core logic

        Parameters
        ----------
        X: 2D np.array with shape (n_instances, n_timepoints)
            panel of univariate time series to cluster

        Returns
        -------
        Numpy_Array: 1D np.array of length n_instances
            Index of the cluster each sample belongs to
        """
        raise NotImplementedError("abstract method")

    def _check_params(self, X: NumpyArray):
        """
        Custom input checking, should raise errors

        Parameters
        ----------
        X: 2D np.array with shape (n_instances, n_timepoints)
        """
        return

    def fit_predict(self, X: NumpyOrDF, y=None) -> NumpyArray:
        """
        clusters time series and returns cluster labels

        Parameters
        ----------
        X: 2D np.array with shape (n_instances, n_timepoints)
           or pd.DataFrame in nested format
            panel of univariate time series to cluster

        y: ignored, exists for API consistency reasons

        Returns
        -------
        Numpy_Array: 1D np.array of length n_instances
            Index of the cluster each sample belongs to
        """
        self.fit(X)
        return self.predict(X)

    def get_fitted_params(self):
        """
        Retrieves fitted parameters of cluster model

        returns
        ----------
        param_dict: dictionary of fitted parameters
        """
        self.check_is_fitted()

        return self._get_fitted_params()

    def _get_fitted_params(self):
        """
        Retrieves fitted parameters of cluster model

            core logic

        returns
        ----------
        param_dict: dictionary of fitted parameters
        """
        raise NotImplementedError


class BaseClusterCenterInitializer:
    """Base Cluster Center Initializer

    Parameters
    ----------
    data_set: Numpy_Array
        Numpy_Array that is the dataset to calculate the centers from

    n_centers: int
        Number of centers to be created

    center_calculator_func: CenterCalculatorFunc, default = None
        Function that is used to calculate new centers

    random_state: NumpyRandomState, default = None
        Generator used to initialise the centers.
    """

    def __init__(
        self,
        data_set: NumpyArray,
        n_centers: int,
        center_calculator_func: CenterCalculatorFunc = None,
        random_state: NumpyRandomState = None,
    ):
        self.data_set: data_set = data_set
        self.n_centers: int = n_centers
        self.center_calculator_func: CenterCalculatorFunc = center_calculator_func
        self.random_state: NumpyRandomState = random_state

    def initialize_centers(self) -> NumpyArray:
        """
        Method used to initialise centers

        Returns
        -------
        Data_Frame
            sktime dataframe containing the starting
            centers
        """
        raise NotImplementedError("abstract method")


class BaseClusterAverage:
    """Base Cluster Average

    Parameters
    ----------
    series: Numpy_Array
        Set of series to generate a average from
    """

    def __init__(self, series: NumpyArray, n_iterations: int = 10):
        self.n_iterations = n_iterations
        self.series = series

    def average(self) -> NumpyArray:
        """
        Method called to find the average for a distance metric

        Returns
        -------
        Numpy_Array
            Created array denoting the average

        n_iterations: int
            Number of iterations to refine the average
        """
        raise NotImplementedError("abstract method")


class BaseApproximate:
    """Base Approximate

    Parameters
    ----------
    series: Numpy_Array
        series to perform approximation on

    """

    def __init__(self, series: NumpyArray):
        self.series = series

    def approximate(self) -> int:
        """
        Method called to get the approximation

        Returns
        -------
        int
            Index position of the approximation in the series
        """
        raise NotImplementedError("abstract method")
