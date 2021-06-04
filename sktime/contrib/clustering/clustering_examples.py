# -*- coding: utf-8 -*-
"""Clustering usage tests and examples"""

from sktime.clustering import (
    TimeSeriesKMeans,
    TimeSeriesKMedoids,
)


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing.
    """
    clusterer1 = TimeSeriesKMeans()
    clusterer2 = TimeSeriesKMedoids()




