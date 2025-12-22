"""Time series Agglomerative Clustering."""

__author__ = ["Muhammad-Rebaal"]
__all__ = ["TimeSeriesAgglomerativeClustering"]


from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sktime.clustering.base import BaseClusterer
from sktime.dists_kernels.base import BasePairwiseTransformerPanel
from sktime.utils.warnings import warn


class TimeSeriesAgglomerativeClustering(BaseClusterer):
    """Agglomerative Clustering for time series.

    Recursive merge of pair of clusters that minimally increases a given linkage
    distance.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    linkage : str, default="average"
        Which linkage criterion to use (e.g., 'ward', 'average', 'complete', 'single').
    distance : str or callable, default="dtw"
        Metric used to compute the linkage.
    distance_params : dict, optional, default=None
        Parameters to be passed to the distance metric.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point.
    linkage_matrix_ : ndarray
        The hierarchical clustering encoded as a linkage matrix.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_mtype": ["pd-multiindex", "numpy3D"],
        "capability:out_of_sample": False,
        "capability:predict": True,
        "capability:predict_proba": False,
    }

    def __init__(
        self,
        n_clusters=2,
        linkage="average",
        distance="dtw",
        distance_params=None,
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance = distance
        self.distance_params = distance_params

        super().__init__()

        from sktime.base._panel.knn import DISTANCES_SUPPORTED

        if isinstance(distance, str) and distance not in DISTANCES_SUPPORTED:
            raise ValueError(
                f"Unrecognised distance measure string: {distance}. "
                f"Allowed values are: {DISTANCES_SUPPORTED}. "
            )

        if isinstance(distance, BasePairwiseTransformerPanel):
            self.clone_tags(
                distance,
                [
                    "capability:multivariate",
                    "capability:unequal_length",
                    "capability:missing_values",
                ],
            )

        if isinstance(distance, str):
            tags_to_set = {
                "X_inner_mtype": "numpy3D",
                "capability:unequal_length": False,
            }
            self.set_tags(**tags_to_set)

    def _fit(self, X, y=None):
        """Fit time series clusterer to training data."""
        self._X = X

        from sktime.dists_kernels.base.adapters._sklearn import _SklearnDistanceAdapter

        dist_adapter = _SklearnDistanceAdapter(
            distance=self.distance,
            distance_params=self.distance_params,
            n_vars=X.shape[1],
            is_equal_length=self._X_metadata["is_equal_length"],
        )
        distmat = dist_adapter._distance(X)

        condensed_dist = squareform(distmat)
        self.linkage_matrix_ = linkage(
            condensed_dist, method=self.linkage, metric="precomputed"
        )

        self.labels_ = (
            fcluster(self.linkage_matrix_, t=self.n_clusters, criterion="maxclust") - 1
        )
        return self

    def _predict(self, X, y=None):
        """Predict the closest cluster each sample in X belongs to."""
        if X is self._X:
            return self.labels_
        else:
            warn(
                f"Predict called with new data. {self.__class__.__name__} will re-fit "
                "on the passed data.",
                obj=self,
            )
            return self.clone().fit(X).labels_
