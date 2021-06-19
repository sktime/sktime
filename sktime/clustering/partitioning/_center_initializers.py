# -*- coding: utf-8 -*-
"""Cluster center initializers"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["ForgyCenterInitializer", "KMeansPlusPlusCenterInitializer"]

import numpy as np

from sktime.clustering.base.base import BaseClusterCenterInitializer
from sktime.clustering.base.base_types import Numpy_Array


class ForgyCenterInitializer(BaseClusterCenterInitializer):
    def __init__(self, data_set: Numpy_Array, n_centers: int):
        """
        Constructor for ForgyCenterInitializer that is used
        to create n centers from a set of series using Forgys
        technique

        Parameters
        ----------
        data_set: Numpy_Array
            Numpy_Array that is the dataset to calculate the centers from

        n_centers: int
            Number of centers to be created

        """
        super(ForgyCenterInitializer, self).__init__(data_set, n_centers)

    def initialize_centers(self) -> Numpy_Array:
        """
        Method called to initialize centers using Forgys
        technique

        Returns
        -------
        Numpy_Array
            numpy array containing the centers
        """
        return self.data_set[
            np.random.choice(self.data_set.shape[0], self.n_centers, replace=False), :
        ]


class KMeansPlusPlusCenterInitializer(BaseClusterCenterInitializer):
    """
    TODO: I need to have a look at this and see if you can use dtw for this
    """

    def __init__(self, data_set: Numpy_Array, n_centers: int):
        """
        Constructor for KMeansPlusPlusCenterInitializer that is used
        to create n centers from a set of series using the kmeans++
        algorithm

        Parameters
        ----------
        data_set: Numpy_Array
            Numpy_Array that is the dataset to calculate the centers from

        n_centers: int
            Number of centers to be created

        """
        super(KMeansPlusPlusCenterInitializer, self).__init__(data_set, n_centers)

    def initialize_centers(self) -> Numpy_Array:
        """
        Returns
        -------
        Numpy_Array
            numpy array containing the centers
        """
        pass


class RandomCenterInitializer(BaseClusterCenterInitializer):
    def __init__(self, data_set: Numpy_Array, n_centers: int):
        """

        Parameters
        ----------
        data_set: Numpy_Array
            Numpy_Array that is the dataset to calculate the centers from

        n_centers: int
            Number of centers to be created

        """
        super(RandomCenterInitializer, self).__init__(data_set, n_centers)

    def initialize_centers(self) -> Numpy_Array:
        """

        Returns
        -------
        Numpy_Array
            numpy array containing the centers

        """
        pass
