# -*- coding: utf-8 -*-
from sktime.base import BaseEstimator
from sktime.clustering.base.base_types import Data_Frame, Numpy_Array
from typing import List, Union, Mapping


class BaseCluster:
    def fit(self, X: Data_Frame) -> None:
        """
        Method that is used to fit the clustering algorithm
        on the dataset X

        Parameters
        ----------
        X: Data_Frame
            sktime data_frame to train the model on
        """
        raise NotImplementedError("abstract method")

    def predict(self, X: Data_Frame) -> List[List[int]]:
        """
        Method used to perform a prediction from the already
        trained clustering algorithm

        Parameters
        ----------
        X: Data_Frame
            sktime data_frame to predict clusters for

        Returns
        -------
        List[List[int]]
            2d array, each sub list contains the indexes that
            belong to that cluster
        """
        raise NotImplementedError("abstract method")


class ClusterMixin(BaseEstimator):
    def fit_predict(self, X: Data_Frame) -> List[List[int]]:
        """
        Method that calls fit and then returns a prediction
        for the value of X

        Parameters
        ----------
        X: Data_Frame
            sktime data_frame containing the values
            to perform clustering algorithm on
        """
        self.fit(X)
        return self.predict(X)


class BaseClusterCenterInitializer:
    def __init__(self, df: Data_Frame, n_centers: int):
        """
        Constructor for BaseClusterCenterInitializer

        Parameters
        ----------
        df: Data_Frame
            sktime data_frame containing values to generate
            centers from

        n_centers: int
            Number of centers to be created
        """
        self.df = df
        self.n_centers = n_centers

    def initialize_centers(self) -> Data_Frame:
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
        self.n_iterations = 10
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
