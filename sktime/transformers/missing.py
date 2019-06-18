import numpy as np

from .base import BaseTransformer

__all__ = ["MissingValuesSegmenter"]
__author__ = ["Piotr Oleśkiewicz"]


class MissingValuesSegmenter(BaseTransformer):
    """
    Missing value transformer.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit transformer, finding positions & lengths of missing value
        intervals.
        """
        raise NotImplementedError

    def __find_missing_intervals(self):
        """
        Find intervals of missing (NaN) values.
        """
        # test data:
        x = np.array([1, 2, np.nan, np.nan, 3, np.nan, np.nan, np.nan, 2, 3, np.nan, 3])
        i = np.isnan(x)
        i_start = np.where(i[:-1] != i[1:])[0][::2] + 1
        i_end = np.where(i[:-1] != i[1:])[0][1::2]
        # lenghts = 1 + i_end - i_start
        return (i_start, i_end)

    def transform(self, X, y=None):
        """
        Transform X.
        """
        raise NotImplementedError
