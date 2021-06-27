# -*- coding: utf-8 -*-
"""Cluster center initializers"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = ["ForgyCenterInitializer", "KMeansPlusPlusCenterInitializer"]

import numpy as np
from sklearn.utils import check_random_state

from sktime.clustering.base import BaseClusterCenterInitializer
from sktime.clustering.base._typing import (
    NumpyArray,
    NumpyRandomState,
    CenterCalculatorFunc,
)


class ForgyCenterInitializer(BaseClusterCenterInitializer):
    """Forgy Center Initializer used to create n centers
    from a set of series using Forgys technique
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
        super(ForgyCenterInitializer, self).__init__(
            data_set, n_centers, center_calculator_func, random_state
        )

    def initialize_centers(self) -> NumpyArray:
        """
        Method called to initialize centers using Forgys
        technique
        Returns
        -------
        Numpy_Array
            numpy array containing the centers
        """
        random_state = check_random_state(self.random_state)
        return self.data_set[
            random_state.choice(self.data_set.shape[0], self.n_centers, replace=False),
            :,
        ]


class RandomCenterInitializer(BaseClusterCenterInitializer):
    """Random Center Initializer used to create n centers
    from randomly assigning each time series in the dataset
    to a random cluster and then taking the approximation
    value
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
        super(RandomCenterInitializer, self).__init__(
            data_set, n_centers, center_calculator_func, random_state
        )

    def initialize_centers(self) -> NumpyArray:
        """
        Method called to initialize centers using Forgys
        technique
        Returns
        -------
        Numpy_Array
            numpy array containing the centers
        """
        if self.center_calculator_func is None:
            raise ValueError(
                "The parameter center_calculator_func must be "
                "specified for this type of center initialisation"
            )
        indexes = self.random_state.choice(
            range(0, self.n_centers), replace=True, size=self.data_set.shape[0]
        )
        temp = np.zeros((self.n_centers, self.data_set.shape[1]))
        for k in range(self.n_centers):
            cluster_values = np.take(self.data_set, np.where(indexes == k), axis=0)[0]
            temp[k] = self.center_calculator_func(cluster_values)
        return np.array(temp, dtype=self.data_set.dtype)


class KMeansPlusPlusCenterInitializer(BaseClusterCenterInitializer):
    """K-means++ center initializer algorithm
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
        super(KMeansPlusPlusCenterInitializer, self).__init__(
            data_set, n_centers, center_calculator_func, random_state
        )

    def initialize_centers(self) -> NumpyArray:
        """
        Method called to initialize centers using Forgys
        technique
        Returns
        -------
        Numpy_Array
            numpy array containing the centers
        """
        return
