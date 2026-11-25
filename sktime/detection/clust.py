"""Cluster Segmentation.

Implementing segmentation using clustering, Read more at
<https://en.wikipedia.org/wiki/Cluster_analysis>_.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans

from sktime.detection.base import BaseDetector

__author__ = ["Ankit-1204"]
__all__ = ["ClusterSegmenter"]


class ClusterSegmenter(BaseDetector):
    """Cluster-based Time Series Segmentation.

    Segments a time series by clustering its time points, using the chosen
    clustering algorithm. Each time point is assigned the label of its cluster;
    segments are the maximal runs of consecutive time points with the same label.

    For multivariate series, time points are clustered as vectors of all variables.

    Parameters
    ----------
    clusterer : sklearn.cluster
        The instance of clustering algorithm used for segmentation.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.cluster import KMeans
    >>> from sktime.detection.clust import ClusterSegmenter
    >>> X = pd.DataFrame({"a": [1.0, 1.1, 0.9, 1.0, 10.0, 10.2, 9.8, 10.1]})
    >>> segmenter = ClusterSegmenter(KMeans(n_clusters=2, random_state=0, n_init=10))
    >>> segmenter.fit_predict(X)
        ilocs  labels
    0  [0, 4)       1
    1  [4, 8)       0
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "Ankit-1204",
        "maintainers": "Ankit-1204",
        # estimator type
        # --------------
        "task": "segmentation",
        "learning_type": "unsupervised",
    }

    def __init__(self, clusterer=None):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below
        self.clusterer = clusterer
        if self.clusterer is None:
            self._clusterer = KMeans()
        else:
            self._clusterer = self.clusterer
        super().__init__()

    def _fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        Y : pd.Series, optional
            ground truth annotations for training if detector is supervised

        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        cloned_clusterer = clone(self._clusterer)
        cloned_clusterer.fit(self._to_2d_array(X))

        self._clusterer_ = cloned_clusterer
        return self

    def _predict(self, X):
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with columns "ilocs" and "labels"
            "ilocs" contains the segments as left-closed ``pd.Interval`` of
            ``iloc`` indices of ``X``, "labels" the cluster label of each segment
        """
        labels = np.asarray(self._clusterer_.predict(self._to_2d_array(X)))

        # segments are maximal runs of equal labels, as half-open iloc intervals
        starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        ends = np.r_[starts[1:], len(labels)]
        intervals = [
            pd.Interval(int(s), int(e), closed="left") for s, e in zip(starts, ends)
        ]
        return pd.DataFrame({"ilocs": intervals, "labels": labels[starts]})

    @staticmethod
    def _to_2d_array(X):
        """Coerce X to a 2D array of time points (rows) by variables (columns)."""
        if isinstance(X, pd.Series):
            X = X.to_frame()
        return np.asarray(X.values, dtype=float).reshape(len(X), -1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for detectors.

        Returns
        -------
        params : dict or list of dict, default = {}

        """
        params1 = {"clusterer": KMeans(n_clusters=2)}
        params2 = {}
        return [params1, params2]
