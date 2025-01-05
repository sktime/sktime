"""Window Clustering Segmentation.

Implementing segmentation using clustering, Read more at
<https://en.wikipedia.org/wiki/Cluster_analysis>_.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.clustering.k_means import TimeSeriesKMeansTslearn
from sktime.detection.base import BaseDetector
from sktime.utils.sklearn import is_sklearn_clusterer

__author__ = ["Ankit-1204"]
__all__ = ["WindowSegmenter"]


def overlapping_window(window_size, step_size, X):
    X_size = len(X)
    n_features = X.shape[1]

    sub_seg = []

    for i in range(0, X_size, step_size):
        end_idx = i + window_size
        segment = X.iloc[i:end_idx].values

        if len(segment) < window_size:
            pad_length = window_size - len(segment)
            pad = np.zeros((pad_length, n_features))
            segment = np.concatenate([segment, pad], axis=0)

        sub_seg.append(segment)

    return np.array(sub_seg).reshape(len(sub_seg), n_features, window_size)


def window_timeseries(window_size, X):
    """Create a list of segments of chosen Window Size for sktime clusterers.

    Parameters
    ----------
    X : Pandas DataFrame
    window_size : Integer

    Returns
    -------
    np.array : 3D numpy array (n_segments, n_features, window_size)
        A 3D array where each segment is reshaped for sktime use.
    """
    X_size = len(X)
    n_features = X.shape[1]

    sub_seg = [
        X.iloc[i : window_size + i].values for i in range(0, X_size, window_size)
    ]

    if len(sub_seg[-1]) < window_size:
        pad_length = window_size - len(sub_seg[-1])
        pad = np.zeros((pad_length, n_features))
        sub_seg[-1] = np.concatenate([sub_seg[-1], pad], axis=0)

    return np.array(sub_seg).reshape(len(sub_seg), n_features, window_size)


def window(window_size, X):
    """Create a list of segments of chosen Window Size with proper padding.

    Parameters
    ----------
    X : Pandas DataFrame
    window_size : Integer

    Returns
    -------
    sub_seg : List of DataFrame
    """
    X_size = len(X)
    sub_seg = [X.iloc[i : window_size + i] for i in range(0, X_size, window_size)]
    if len(sub_seg[-1]) < window_size:
        re = window_size - len(sub_seg[-1])
        remainder = pd.DataFrame(0, index=range(re), columns=X.columns)
        sub_seg[-1] = pd.concat([sub_seg[-1], remainder], ignore_index=True)
    return sub_seg


def flattenSegments(sub_seg):
    """Ensure that the function supports multivariate series by Flattening each segment.

    Parameters
    ----------
        sub_seg : List of DataFrame

    Returns
    -------
        np.array(flat) : Numpy Array
    """
    flat = [i.values.flatten() for i in sub_seg]
    return np.array(flat)


def finalLabels(labels, window_size, X):
    """Convert segment labels to individual time point labels.

    Parameters
    ----------
        X :Pandas DataFrame
        window_size : Integer
        labels : List

    Returns
    -------
        np.array(flabel) : Numpy Array
    """
    X_size = len(X)
    flabel = [labels[i // window_size] for i in range(X_size)]
    return np.array(flabel)


def overlap_final_label(labels, window_size, step_size, X):
    time_point_labels = [[] for _ in range(len(X))]

    for i in range(len(labels)):
        start_ind = i * step_size
        end_ind = start_ind + window_size
        if end_ind > len(X):
            end_ind = len(X)
        for j in range(start_ind, end_ind):
            time_point_labels[j].append(int(labels[i]))
    return time_point_labels


def aggregate_labels(flabel):
    aggr_labels = []
    for labels in flabel:
        most_freq = Counter(labels).most_common(1)[0][0]
        aggr_labels.append(most_freq)
    return aggr_labels


class WindowSegmenter(BaseDetector):
    """Window-based Time Series Segmentation via Clustering.

    In this we get overlapping and non overlapping subseries using a Sliding window.
    After that we run a clustering algorithm of our choosing to segment the
    time series.

    todo: write docstring, describing your custom forecaster

    Parameters
    ----------
    clusterer : sklearn.cluster
        The instance of clustering algorithm used for segmentation.
    window_size : Integer
        The size of Sliding Window

    Examples
    --------
    >>> from sktime.detection.wclust import WindowSegmenter
    >>> from sktime.datasets import load_gunpoint
    >>> X, y = load_gunpoint()
    >>> clusterer = KMeans()
    >>> segmenter = ClusterSegmenter(clusterer, 3)
    >>> segmenter._fit(X)
    >>> segment_labels = segmenter._predict(X)
    """

    _tags = {
        "task": "segmentation",
        "learning_type": "unsupervised",
    }

    def __init__(
        self,
        clusterer=None,
        window_size=1,
        overlap=False,
        step_size=1,
        return_segments=True,
    ):
        self.clusterer = clusterer
        self._clusterer_ = clusterer
        self.window_size = window_size
        self._window_size = window_size
        self.overlap = overlap
        self.step_size = step_size
        self.return_segments = return_segments
        if self.clusterer is None:
            self._clusterer = TimeSeriesKMeansTslearn()
        else:
            self._clusterer = self.clusterer
        super().__init__()

    def _fit(self, X, y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        y : pd.Series, optional
            ground truth detections for training if annotator is supervised

        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        if isinstance(X, pd.Series):
            X = X.to_frame(X)
        if self.overlap:
            win_x = overlapping_window(self._window_size, self.step_size, X)
            cloned_clusterer = clone(self._clusterer)
            cloned_clusterer.fit(win_x)
            self._clusterer_ = cloned_clusterer
        else:
            if is_sklearn_clusterer(self._clusterer):
                win_x = window(self._window_size, X)
                seg = flattenSegments(win_x)
                cloned_clusterer = clone(self._clusterer)
                cloned_clusterer.fit(seg)
            else:
                win_x = window_timeseries(self._window_size, X)
                cloned_clusterer = clone(self._clusterer)
                cloned_clusterer.fit(win_x)
            self._clusterer_ = cloned_clusterer

        return self

    def _predict(self, X):
        """Create detections on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - detections for sequence X
            exact format depends on detection type
        """
        if isinstance(X, pd.Series):
            X = X.to_frame(X)

        self.n_features, self.n_timepoints = X.shape
        if self.overlap:
            win_x = overlapping_window(self._window_size, self.step_size, X)
            labels = self._clusterer_.predict(win_x)
            flabel = overlap_final_label(labels, self._window_size, self.step_size, X)
            if self.return_segments:
                flabel = pd.Series(flabel, index=X.index)
            else:
                aggr_labels = aggregate_labels(flabel)
                return pd.Series(aggr_labels, index=X.index)
        else:
            if is_sklearn_clusterer(self._clusterer_):
                win_x = window(self._window_size, X)
                sub = flattenSegments(win_x)
                labels = self._clusterer_.predict(sub)
            else:
                win_x = window_timeseries(self._window_size, X)
                labels = self._clusterer_.predict(win_x)
            flabel = finalLabels(labels, self._window_size, X)
            flabel = pd.Series(flabel.flatten(), index=X.index)

        return flabel

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
        """
        params1 = {"clusterer": TimeSeriesKMeansTslearn(n_clusters=2), "window_size": 2}
        params2 = {}
        return [params1, params2]
