"""Cluster Segmentation.

Implementing segmentation using clustering, Read more at
<https://en.wikipedia.org/wiki/Cluster_analysis>_.
"""

import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans

from sktime.detection.base import BaseDetector

__author__ = ["Ankit-1204"]
__all__ = ["ClusterSegmenter"]


class ClusterSegmenter(BaseDetector):
    """Cluster-based Time Series Segmentation.

    time series segmentation using clustering is simple task. This detector
    segments time series data into distinct segments based on similarity, identified
    using the chosen clustering algorithm.

    The wrapped clusterer predicts one label per time point. ``ClusterSegmenter``
    then groups consecutive equal labels into sparse labelled segments, which is the
    contract expected from segmentation detectors in ``sktime``.

    Parameters
    ----------
    clusterer : sklearn.cluster
        The instance of clustering algorithm used for segmentation.
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
        "capability:multivariate": False,
        # CI and test flags
        # -----------------
        "tests:skip_by_name": ["test_non_state_changing_method_contract"],
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
        if isinstance(X, pd.Series):
            X = X.to_frame()
        cloned_clusterer = clone(self._clusterer)
        X_flat = X.values.reshape(-1, 1)
        cloned_clusterer.fit(X_flat)

        self._clusterer_ = cloned_clusterer
        return self

    def _predict(self, X):
        """Create sparse segmentation output on test/deployment data.

        The clustering model predicts one label per time point. This method then
        converts the dense label sequence into the sparse segment format expected by
        ``BaseDetector`` for ``task="segmentation"``.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        pd.DataFrame
            Segments in sparse format with columns ``"ilocs"`` and ``"labels"``.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame()

        X_flat = X.values.reshape(-1, 1)
        labels = self._clusterer_.predict(X_flat)
        labels = pd.Series(labels, index=pd.RangeIndex(len(labels)), dtype="int64")
        return self._labels_to_segments(labels)

    @staticmethod
    def _labels_to_segments(labels):
        """Convert dense per-timepoint labels to sparse labelled segments."""
        if len(labels) == 0:
            return BaseDetector._empty_segments().assign(
                labels=pd.Series(dtype="int64")
            )

        dense = labels.reset_index(drop=True)
        start = 0
        intervals = []
        segment_labels = []

        for i in range(1, len(dense)):
            if dense.iloc[i] != dense.iloc[i - 1]:
                intervals.append(pd.Interval(start, i, closed="left"))
                segment_labels.append(dense.iloc[start])
                start = i

        intervals.append(pd.Interval(start, len(dense), closed="left"))
        segment_labels.append(dense.iloc[start])

        return pd.DataFrame(
            {
                "ilocs": pd.IntervalIndex(intervals),
                "labels": pd.Series(segment_labels, dtype="int64"),
            }
        )

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
        params1 = {"clusterer": KMeans(n_clusters=2, random_state=0)}
        params2 = {"clusterer": KMeans(n_clusters=3, random_state=0)}
        return [params1, params2]
