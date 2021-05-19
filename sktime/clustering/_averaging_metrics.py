# -*- coding: utf-8 -*-
from sktime.clustering.base.base import BaseClusterAverage


class BarycenterAveraging(BaseClusterAverage):
    """
    Implementations based off:
    https://blog.acolyer.org/2016/05/13/
    dynamic-time-warping-averaging-of-time-series-allows-faster
    -and-more-accurate-classification/
    """

    @staticmethod
    def average(series, iterations=100):
        pass
