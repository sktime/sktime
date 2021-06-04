# -*- coding: utf-8 -*-
from sktime.clustering.base.base import BaseClusterCenterInitializer
from sktime.clustering.base.base_types import Data_Frame

__author__ = "Christopher Holder"


class RandomCenterInitializer(BaseClusterCenterInitializer):
    def __init__(self, df: Data_Frame, n_centers: int):
        """
        Constructor for RandomCenterInitialiser that is used
        to create n centers from a set of series randomly

        Parameters
        ----------
        df: Data_Frame
            sktime data_frame containing values to generate
            centers from

        n_centers: int
            Number of centers to be created

        """
        super(RandomCenterInitializer, self).__init__(df, n_centers)

    def initialize_centers(self) -> Data_Frame:
        """
        Method called to intialize centers randomly

        Returns
        -------
        Data_Frame
            sktime data_frame containing the centers
        """
        return self.df.sample(n=self.n_centers)


class KMeansPlusPlusCenterInitializer(BaseClusterCenterInitializer):
    """
    TODO: I need to have a look at this and see if you can use dtw for this
    """

    def __init__(self, df: Data_Frame, n_centers: int):
        """
        Constructor for RandomCenterInitialiser that is used
        to create n centers from a set of series using the kmeans++
        algorithm

        Parameters
        ----------
        df: Data_Frame
            sktime data_frame containing values to generate
            centers from

        n_centers: int
            Number of centers to be created

        """
        super(KMeansPlusPlusCenterInitializer, self).__init__(df, n_centers)

    def initialize_centers(self) -> Data_Frame:
        """
        Method called to intialize centers randomly

        Returns
        -------
        Data_Frame
            sktime data_frame containing the centers
        """
        pass
