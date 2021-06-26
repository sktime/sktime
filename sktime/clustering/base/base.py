# -*- coding: utf-8 -*-
"""Base classes for clustering."""
__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = [
    "BaseCluster",
    "ClusterMixin",
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


class BaseCluster(BaseEstimator):
    """Base Clusterer"""

    def __init__(self):
        super(BaseCluster, self).__init__()

    def fit(self, X: NumpyOrDF, y: NumpyOrDF = None):
        """
        Method that is used to fit the clustering algorithm
        on the dataset X

        Parameters
        ----------
        X: Numpy array or Dataframe
            sktime data_frame or numpy array to train the model on

        y: Numpy array of Dataframe, default = None
            sktime data_frame or numpy array that is the labels for training.
            Unlikely to be used for clustering but kept for consistency

        Returns
        -------
        self
            Fitted estimator
        """
        if isinstance(X, pd.DataFrame):
            X = from_nested_to_2d_array(X, return_numpy=True)

        self._check_params(X)
        self._fit(X)

        self._is_fitted = True
        return self

    def predict(self, X: NumpyOrDF) -> NumpyArray:
        """
        Method used to perform a prediction from the already
        trained clustering algorithm

        Parameters
        ----------
        X: Numpy array or Dataframe
            sktime data_frame or numpy array to predict
            cluster for

        Returns
        -------
        Numpy_Array: np.array
            Index of the cluster each sample belongs to
        """
        self.check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = from_nested_to_2d_array(X, return_numpy=True)

        return self._predict(X)

    def _fit(self, X: NumpyArray, y: NumpyArray = None):
        """
        Method that contains the core logic to fit a cluster
        to training data

        Parameters
        ----------
        X: Numpy array
            Numpy array to train the model on

        y: Numpy array, default = None
            Numpy array that is the labels for training.
            Unlikely to be used for clustering but kept for consistency

        Returns
        -------
        self
            Fitted estimator
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X: NumpyArray) -> NumpyArray:
        """
        Method that is used
        """
        raise NotImplementedError("abstract method")

    def _check_params(self, X: NumpyArray):
        """
        Method used to check the parameters passed

        Parameters
        ----------
        X: Numpy_Array
            Dataset to be validate parameters against
        """
        return


class ClusterMixin:
    """ClustererMixin"""

    def fit_predict(self, X: NumpyOrDF) -> NumpyArray:
        """
        Method that calls fit and then returns a prediction
        for the value of X

        Parameters
        ----------
        X: Numpy array or Dataframe
            sktime data_frame or numpy array containing the values
            to perform clustering algorithm on
        """
        self.fit(X)
        return self.predict(X)


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
