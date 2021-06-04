# -*- coding: utf-8 -*-
from sktime.clustering.base.base import BaseClusterCenterInitializer
from sktime.clustering.base.base_types import Data_Frame


class RandomBaseClusterCenterInitializer(BaseClusterCenterInitializer):
    @staticmethod
    def initialize_centers(df: Data_Frame, n_centers: int) -> Data_Frame:
        return df.sample(n=n_centers)


class KMeansPlusPlusInitializerBaseCluster(BaseClusterCenterInitializer):
    @staticmethod
    def initialize_centers(df: Data_Frame, n_centers: int) -> Data_Frame:
        pass
