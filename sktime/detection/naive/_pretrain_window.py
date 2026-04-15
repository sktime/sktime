import pandas as pd
import numpy as np
from sktime.detection.supervised import BaseSupervisedDetector


class NaivePretrainWindowDetector(BaseSupervisedDetector):
    """todo: write! We are only dealing with change points first, no segments.
    Maybe it might be confusing to the user that X and y are ordered in the opposite way than in the rest of sktime, but it is more intuitive to have the "training data" as y and the "test data" as X, as for usual supervised learning
    """
    def __init__(self, window_length=10):
        self.window_length = window_length
        self.in_window_mean = None
        self._in_window_counts = 0
        self.out_window_mean = None
        self._out_window_counts = 0
        super().__init__()

    def _pretrain(self, X: pd.DataFrame, y: pd.Series):
        index_ranges = np.column_stack((y.values, y.values + self.window_length))
        idx_arr = np.concatenate([np.arange(s, e) for s, e in index_ranges])
        self.in_window_values = X.iloc[idx_arr].mean()
        self._in_window_counts = len(idx_arr)
        self.out_window_values = X.iloc[~idx_arr].mean()
        self._out_window_counts = len(X) - self._in_window_counts
        return self

    def _predict(self, fh=None, X=None):
        pass

    # def 



