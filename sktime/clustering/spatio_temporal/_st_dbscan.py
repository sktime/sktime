"""Spatiotemporal DBSCAN."""

__author__ = ["eren-ck", "vagechirkov"]

import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sktime.clustering import BaseClusterer
from sktime.datatypes import update_data
from sktime.utils.warnings import warn


class STDBSCAN(BaseClusterer):
    """Spatio-temporal DBSCAN clustering.

    Implementation of STDBSCAN by Birant et al [1].
    Partially based on the implementation of Cakmak et al [3].

    Clusters data based on specified spatial and temporal proximity thresholds.

    Assumes that all variables are spatial coordinates.

    Parameters
    ----------
    eps1 : float, default=0.5
        Maximum spatial distance for points to be considered related.
    eps2 : float, default=10
        Maximum temporal distance for points to be considered related [1].
    min_samples : int, default=5
        Minimum number of samples to form a core point.
    metric : str, default='euclidean'
        Distance metric to use; options include 'euclidean', 'manhattan',
        'chebyshev', etc.
    sparse_matrix_threshold : int, default=20_000
        Sets the limit on the number of samples for which the algorithm can
        efficiently compute distances with a full matrix approach. Datasets
        exceeding this threshold will be handled using sparse matrix methods.
    frame_size : float or None, default=None
        If not None the dataset is split into frames [2, 3];
        The frame_size is the number of time points in a frame.
    frame_overlap : float or None, default=eps2
        If frame_size is set - there will be an overlap between the frames
        to merge the clusters afterward [2, 3]; Only used if frame_size is not None.
    n_jobs : int or None, default=-1
        Number of parallel jobs for distance computation; -1 uses all cores.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point; noise is labeled as -1.

    References
    ----------
    .. [1] Birant, D., & Kut, A. "ST-DBSCAN: An algorithm for clustering
       spatial-temporal data." Data Knowl. Eng., vol. 60, no. 1, pp. 208-221, Jan. 2007,
       doi: [10.1016/j.datak.2006.01.013](https://doi.org/10.1016/j.datak.2006.01.013).
    .. [2] Peca, I., Fuchs, G., Vrotsou, K., Andrienko, N., and Andrienko, G.,
       "Scalable Cluster Analysis of Spatial Events" 2012, The Eurographics Association
       doi: [10.2312/PE/EUROVAST/EUROVA12/019-023](https://doi.org/10.2312/PE/EUROVAST/EUROVA12/019-023).
    .. [3] Cakmak, E., Plank, M., Calovi, D. S., Jordan, A., & Keim, D. "Spatio-temporal
       clustering benchmark for collective animal behavior." ACM, Nov. 2021, pp. 5-8.
       doi: [10.1145/3486637.3489487](https://doi.org/10.1145/3486637.3489487).

    Examples
    --------
    >>> from sktime.clustering.spatio_temporal import STDBSCAN
    >>> from sktime.clustering.utils.toy_data_generation._make_moving_blobs import (
    ... make_moving_blobs)
    >>> X, y_true = make_moving_blobs(n_times=20)
    >>> st_dbscan = STDBSCAN(
    ...     eps1=0.5, eps2=3, min_samples=5, metric="euclidean", n_jobs=-1
    ... )
    >>> st_dbscan.fit(X)
    >>> predicted_labels = st_dbscan.labels_
    """

    _tags = {
        "maintainers": "vagechirkov",
        "authors": ["eren-ck", "vagechirkov"],
        "python_dependencies": ["scipy", "scikit-learn"],
        "X_inner_mtype": "pd-multiindex",
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:multithreading": True,
        "capability:predict": True,
        "capability:predict_proba": False,
        "capability:out_of_sample": True,
    }

    DELEGATED_FITTED_PARAMS = ["core_sample_indices_", "components_ ", "labels_"]

    def __init__(
        self,
        eps1=0.5,
        eps2=10,
        min_samples=5,
        metric="euclidean",
        sparse_matrix_threshold=20_000,
        frame_size=None,
        frame_overlap=None,
        n_jobs=-1,
    ):
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.metric = metric
        self.sparse_matrix_threshold = sparse_matrix_threshold
        self.frame_size = frame_size
        self.frame_overlap = frame_overlap
        self.n_jobs = n_jobs
        self.dbscan_ = None

        super().__init__()

    def _fit(self, X, y=None):
        """
        Apply the ST-DBSCAN algorithm to cluster spatiotemporal data.

        Parameters
        ----------
        X : sktime compatible Panel data container, mtype="pd-multiindex".
            The first index level is time, the second is object (agent) ID.
            Each row represents spatial coordinates of an object at a given time.
            Example of X:
                index object | index time | coordinates x | coordinates y
                -------------|------------|---------------|---------------
                0            | 0          | 0.1           | 0.2
                1            | 0          | 0.3           | 0.4
                0            | 1          | 0.2           | 0.3
                1            | 1          | 0.4           | 0.5
                2            | 1          | 0.5           | 0.6
                0            | 2          | 0.3           | 0.4
                3            | 2          | 0.6           | 0.7
        y : ignored, exists for API consistency reasons

        Returns
        -------
        self :
            Fitted instance with cluster labels.
        """
        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError("eps1, eps2, min_samples  must be positive")

        # make sure that X is sorted by time
        X = X.copy()
        X.sort_index(inplace=True, ascending=True, level=1)

        self._X = X

        if self.frame_size is not None:
            if self.frame_overlap is None:
                self.frame_overlap = self.eps2
            if (
                not self.frame_size > 0.0
                or not self.frame_overlap > 0.0
                or self.frame_size < self.frame_overlap
            ):
                raise ValueError("frame_size, frame_overlap not correctly configured.")
            return self._fit_frame_split(X)
        else:
            self._fit_one_frame(X)
            for key in self.DELEGATED_FITTED_PARAMS:
                if hasattr(self.dbscan_, key):
                    setattr(self, key, getattr(self.dbscan_, key))

        return self

    def _fit_one_frame(self, X):
        if len(X) < self.sparse_matrix_threshold:
            self._fit_dense(X)
        else:
            self._fit_sparse(X)

    def _fit_dense(self, X):
        """Fit the dense distance matrix version of the ST-DBSCAN algorithm."""
        n, m = X.values.shape
        time_index = X.index.get_level_values(1).to_numpy()

        # Compute squared form Distance Matrix
        time_dist = pdist(time_index.reshape(n, 1), metric=self.metric)
        spatial_dist = pdist(X.values, metric=self.metric)

        # filter the spatial_dist matrix using the time_dist
        dist = np.where(time_dist <= self.eps2, spatial_dist, 2 * self.eps1)

        self.dbscan_ = DBSCAN(
            eps=self.eps1, min_samples=self.min_samples, metric="precomputed"
        )
        self.dbscan_.fit(squareform(dist))

    def _fit_sparse(self, X):
        """Fit the sparse distance matrix version of the ST-DBSCAN algorithm."""
        n, m = X.values.shape
        time_index = X.index.get_level_values(1).to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # compute with sparse matrices
            # Compute sparse matrix for spatial distance
            nn_spatial = NearestNeighbors(
                metric=self.metric, radius=self.eps1, n_jobs=self.n_jobs
            )
            nn_spatial.fit(X.values)
            euc_sp = nn_spatial.radius_neighbors_graph(X.values, mode="distance")

            # Compute sparse matrix for temporal distance
            nn_time = NearestNeighbors(
                metric=self.metric, radius=self.eps2, n_jobs=self.n_jobs
            )
            nn_time.fit(time_index.reshape(n, 1))
            time_sp = nn_time.radius_neighbors_graph(
                time_index.reshape(n, 1), mode="distance"
            )

            # combine both sparse matrices and filter by time distance matrix
            row = time_sp.nonzero()[0]
            column = time_sp.nonzero()[1]
            v = np.array(euc_sp[row, column])[0]

            # create sparse distance matrix
            dist_sp = coo_matrix((v, (row, column)), shape=(n, n))
            dist_sp = dist_sp.tocsc()
            dist_sp.eliminate_zeros()

            self.dbscan_ = DBSCAN(
                eps=self.eps1, min_samples=self.min_samples, metric="precomputed"
            )
            self.dbscan_.fit(dist_sp)

    def _fit_frame_split(self, X):
        """Apply the ST-DBSCAN algorithm with splitting it into frames.

        References
        ----------
        .. [1] I. Peca, G. Fuchs, K. Vrotsou, N. Andrienko, and G. Andrienko, “Scalable
           Cluster Analysis of Spatial Events,” 2012, The Eurographics Association. doi:
           [10.2312/PE/EUROVAST/EUROVA12/019-023](https://doi.org/10.2312/PE/EUROVAST/EUROVA12/019-023).
        """
        # unique time points
        time_index = X.index.get_level_values(1).to_numpy()
        time = np.unique(time_index)
        labels = None
        right_overlap = 0
        step = int(self.frame_size - self.frame_overlap + 1)

        for i in range(0, len(time), step):
            for period in [time[i : i + self.frame_size]]:
                frame = X[np.isin(time_index, period)]

                self._fit_one_frame(frame)

                # Match the labels in the overlapped zone.
                # Objects in the second frame are relabeled
                # to match the cluster ID from the first frame.
                if labels is None:
                    labels = self.dbscan_.labels_
                else:
                    frame_1_overlap_labels = labels[len(labels) - right_overlap :]
                    frame_2_overlap_labels = self.dbscan_.labels_[0:right_overlap]

                    mapper = {}
                    for i1, i2 in zip(frame_1_overlap_labels, frame_2_overlap_labels):
                        mapper[i2] = i1
                    mapper[-1] = -1  # avoiding outliers being mapped to cluster

                    # clusters without overlapping points are given new cluster
                    ignore_clusters = set(self.dbscan_.labels_) - set(
                        frame_2_overlap_labels
                    )
                    # recode them to new cluster value
                    if -1 in labels:
                        labels_counter = len(set(labels)) - 1
                    else:
                        labels_counter = len(set(labels))

                    for j in ignore_clusters:
                        mapper[j] = labels_counter
                        labels_counter += 1

                    # Relabel objects in the second frame to match the cluster ID from
                    # the first frame.
                    # Assign new clusters to objects in clusters without overlap
                    new_labels = np.array([mapper[j] for j in self.dbscan_.labels_])

                    # delete the right overlap
                    labels = labels[0 : len(labels) - right_overlap]
                    # change the labels of the new clustering and concat
                    labels = np.concatenate((labels, new_labels))

                right_overlap = len(
                    X.values[np.isin(time_index, period[-self.frame_overlap + 1 :])]
                )
        self.labels_ = labels

    def _predict(self, X, y=None):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : panel of time series, any sklearn Panel mtype
            Time series instances to predict cluster indexes for
        y: ignored, exists for API consistency reasons

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to
        """
        # if X is the same as seen in _fit, simply return the labels
        if X is self._X:
            return self.labels_
        else:
            all_X = update_data(X=self._X, X_new=X)
            warn(
                "sklearn and sktime DBSCAN estimators do not support different X "
                "in fit and predict, but a new X was passed in predict. "
                "Therefore, a clone of STDBSCAN will be fit, and results "
                "returned, without updating the state of the fitted estimator.",
                obj=self,
            )
            return self.clone().fit(all_X).labels_

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for clusterers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        param1 = {}

        param2 = {
            "eps1": 0.3,
            "eps2": 5,
            "min_samples": 3,
            "metric": "euclidean",
            "n_jobs": 1,
        }
        param3 = {
            "eps1": 0.5,
            "eps2": 10,
            "min_samples": 2,
            "metric": "manhattan",
            "n_jobs": 2,
        }

        return [param1, param2, param3]
