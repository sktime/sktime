# -*- coding: utf-8 -*-
"""Base classes for clustering."""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = [
    "BaseCluster",
    "ClusterMixin",
    "BaseClusterCenterInitializer",
    "BaseClusterAverage",
    "BaseApproximate",
    "Init_Algo",
    "Averaging_Algo",
    "Averaging_Algo_Dict",
    "Init_Algo_Dict",
]

import pandas as pd
from sktime.base import BaseEstimator
from sktime.utils.data_processing import from_nested_to_2d_array
from sktime.clustering.base.base_types import Numpy_Array, Numpy_Or_DF
from typing import Union, Mapping


class BaseCluster(BaseEstimator):
    def __init__(self):
        self._is_fitted = False
        super(BaseCluster, self).__init__()

    def fit(self, X: Numpy_Or_DF) -> None:
        """
        Method that is used to fit the clustering algorithm
        on the dataset X

        Parameters
        ----------
        X: Numpy array or Dataframe
            sktime data_frame or numpy array to train the model on

        Returns
        -------
        self
            Fitted estimator
        """
        self._is_fitted = False

        if isinstance(X, pd.DataFrame):
            X = from_nested_to_2d_array(X, return_numpy=True)

        self._check_params(X)
        self._fit(X)

        self._is_fitted = True
        return self

    def predict(self, X: Numpy_Or_DF) -> Numpy_Array:
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

    def _fit(self, X: Numpy_Array) -> None:
        raise NotImplementedError("abstract method")

    def _predict(self, X: Numpy_Array) -> Numpy_Array:
        raise NotImplementedError("abstract method")

    def _check_params(self, X: Numpy_Array):
        """
        Method used to check the parameters passed

        Parameters
        ----------
        X: Numpy_Array
            Dataset to be validate parameters against
        """
        return


class ClusterMixin:
    def fit_predict(self, X: Numpy_Or_DF) -> Numpy_Array:
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
    def __init__(self, data_set: Numpy_Array, n_centers: int):
        """
        Constructor for BaseClusterCenterInitializer

        Parameters
        ----------
        data_set: Numpy_Array
            Numpy_Array that is the dataset to calculate the centers from

        n_centers: int
            Number of centers to be created
        """
        self.data_set = data_set
        self.n_centers = n_centers

    def initialize_centers(self) -> Numpy_Array:
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
    def __init__(self, series: Numpy_Array, n_iterations: int = 10):
        """
        Constructor for BaseClusterAverage

        Parameters
        ----------
        series: Numpy_Array
            Set of series to generate a average from
        """
        self.n_iterations = n_iterations
        self.series = series

    def average(self) -> Numpy_Array:
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
    def __init__(self, series: Numpy_Array):
        """
        Constructor for BaseAprroximate

        Parameters
        ----------
        series: Numpy_Array
            series to perform approximation on

        """
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


Init_Algo = Union[str, BaseClusterCenterInitializer]
Init_Algo_Dict = Mapping[str, BaseClusterCenterInitializer]

Averaging_Algo = Union[str, BaseClusterAverage]
Averaging_Algo_Dict = Mapping[str, Averaging_Algo]
