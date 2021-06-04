# -*- coding: utf-8 -*-
"""Barycentre Averaging"""

__author__ = "Chris Holder"
__all__ = ["BarycenterAveraging"]

from sktime.clustering.base.base import BaseClusterAverage


class BarycenterAveraging(BaseClusterAverage):
    """
    Implementations based on:
    https://blog.acolyer.org/2016/05/13/
    dynamic-time-warping-averaging-of-time-series-allows-faster
    -and-more-accurate-classification/
    """

    @staticmethod
    def average(series, iterations=100):
        pass
