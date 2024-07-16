"""Cluster Segmentation.

Implementing segmentation using clustering, Read more at
<https://en.wikipedia.org/wiki/Cluster_analysis>_.
"""

import pandas as pd
from sklearn.cluster import KMeans

from sktime.annotation.base import BaseSeriesAnnotator

__author__ = ["Ankit-1204"]
__all__ = ["ClusterSegmenter"]


class ClusterSegmenter(BaseSeriesAnnotator):
    """Cluster-based Time Series Segmentation.

    time series segmentation using clustering is simple task. This annotator
    segments time series data into distinct segments based on similarity, identified
    using the choosen clustering algorithm.

    Parameters
    ----------
    clusterer : sklearn.cluster
        The instance of clustering algorithm used for segmentation.
    n_clusters : int, default=3
        The number of clusters to form

    Examples
    --------
    >>> from sktime.annotation.cluster import ClusterSegmenter
    >>> from sktime.datasets import load_gunpoint
    >>> X, y = load_gunpoint()
    >>> clusterer = KMeans(n_clusters=2)
    >>> segmenter = ClusterSegmenter(clusterer)
    >>> segmenter.fit(X)
    >>> segment_labels = segmenter.predict(X)

    """

    _tags = {
        "task": "segmentation",
        "learning_type": "unsupervised",
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, clusterer=None, n_clusters=3):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.clusterer = (
            clusterer if clusterer is not None else KMeans(n_clusters=n_clusters)
        )
        self.n_clusters = n_clusters

        super().__init__()

    def fit(self, X, Y=None):
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
        self.n_instances, self.n_timepoints = X.shape
        X_flat = X.values.reshape(-1, 1)
        self.clusterer.fit(X_flat)
        return self

    def predict(self, X):
        """Create annotations on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        X_flat = X.values.reshape(-1, 1)
        labels = self.clusterer.predict(X_flat)
        labels = labels.reshape(self.n_instances, self.n_timepoints)
        return pd.DataFrame(labels, index=X.index)

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
        return {"clusterer": KMeans(n_clusters=2), "n_clusters": 2}
