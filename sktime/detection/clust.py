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

    time series segmentation using clustering is simple task. This annotator
    segments time series data into distinct segments based on similarity, identified
    using the choosen clustering algorithm.

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
            ground truth annotations for training if annotator is supervised

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
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        if isinstance(X, pd.Series):
            X = X.to_frame()
        self.n_instances, self.n_timepoints = X.shape
        X_flat = X.values.reshape(-1, 1)
        labels = self._clusterer_.predict(X_flat)
        labels = labels.reshape(self.n_instances, self.n_timepoints)
        if self.n_instances == 1:
            return pd.Series(labels.flatten(), index=X.index)

        return pd.Series(labels.flatten(), index=X.index.repeat(self.n_timepoints))

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
        params1 = {"clusterer": KMeans(n_clusters=2)}
        params2 = {}
        return [params1, params2]
