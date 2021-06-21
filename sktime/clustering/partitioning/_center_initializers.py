# -*- coding: utf-8 -*-
"""Cluster center initializers"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["ForgyCenterInitializer", "KMeansPlusPlusCenterInitializer"]

import numpy as np

from sktime.clustering.base.base import BaseClusterCenterInitializer
from sktime.clustering.base._typing import NumpyArray, NumpyRandomState


class ForgyCenterInitializer(BaseClusterCenterInitializer):
    """Forgy Center Initializer used to create n centers
    from a set of series using Forgys technique

    Parameters
    ----------
    data_set: Numpy_Array
        Numpy_Array that is the dataset to calculate the centers from

    n_centers: int
        Number of centers to be created

    random_state: NumpyRandomState, default = None
        Generator used to initialise the centers.
    """

    def __init__(
        self,
        data_set: NumpyArray,
        n_centers: int,
        random_state: NumpyRandomState = None,
    ):
        super(ForgyCenterInitializer, self).__init__(data_set, n_centers, random_state)

    def initialize_centers(self) -> NumpyArray:
        """
        Method called to initialize centers using Forgys
        technique

        Returns
        -------
        Numpy_Array
            numpy array containing the centers
        """
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        return self.data_set[
            self.random_state.choice(
                self.data_set.shape[0], self.n_centers, replace=False
            ),
            :,
        ]


class KMeansPlusPlusCenterInitializer(BaseClusterCenterInitializer):
    """K-Means++ Center Initializers that is used to create n
    centers from a set of series using the kmeans++ algorithm

    Parameters
    ----------
    data_set: Numpy_Array
        Numpy_Array that is the dataset to calculate the centers from

    n_centers: int
        Number of centers to be created

    """

    def __init__(self, data_set: NumpyArray, n_centers: int):
        super(KMeansPlusPlusCenterInitializer, self).__init__(data_set, n_centers)

    def initialize_centers(self) -> NumpyArray:
        """
        Returns
        -------
        Numpy_Array
            numpy array containing the centers
        """
        pass


class RandomCenterInitializer(BaseClusterCenterInitializer):
    """Random Center Initializer that is used to create n
    centers from a set of series using the random center
    initializer algorithm

    Parameters
    ----------
    data_set: Numpy_Array
        Numpy_Array that is the dataset to calculate the centers from

    n_centers: int
        Number of centers to be created

    """

    def __init__(self, data_set: NumpyArray, n_centers: int):
        super(RandomCenterInitializer, self).__init__(data_set, n_centers)

    def initialize_centers(self) -> NumpyArray:
        """

        Returns
        -------
        Numpy_Array
            numpy array containing the centers

        """
        pass
