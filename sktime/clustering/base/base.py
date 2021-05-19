# -*- coding: utf-8 -*-
from sktime.base import BaseEstimator
from sktime.clustering.base.base_types import Data_Frame
from typing import List, Union, Mapping


class BaseCluster:
    def fit(self, X: Data_Frame) -> None:
        raise NotImplementedError("abstract method")

    def predict(self, X: Data_Frame) -> List[List[int]]:
        raise NotImplementedError("abstract method")


class ClusterMixin(BaseEstimator):
    def fit_predict(self, X: Data_Frame) -> List[List[int]]:
        self.fit(X)
        return self.predict(X)


class BaseClusterCenterInitializer:
    @staticmethod
    def initialize_centers(df: Data_Frame, n_centers: int) -> Data_Frame:
        raise NotImplementedError("abstract method")


class BaseClusterAverage:
    @staticmethod
    def average(series, iterations=100):
        """
        Method called to find the average for a distance metric

        Parameters
        ----------
        series:
            The set of sequences to average

        iterations: int default = 100
            The number of iterations
        """
        raise NotImplementedError("abstract method")


Init_Algo = Union[str, BaseClusterCenterInitializer]
Init_Algo_Dict = Mapping[str, BaseClusterCenterInitializer]
