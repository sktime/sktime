"""Time series Kvisibility"""

__author__ = ["seigpe"]

import numpy as np
from sklearn.cluster import KMeans
from sktime.clustering.base import BaseClusterer
from sktime.datatypes import update_data
from sktime.dists_kernels.base import BasePairwiseTransformerPanel
from sktime.utils.warnings import warn


class TimeSeriesKvisibility(BaseClusterer):
    """Kvisibility for time series distances.

    Interface to Kvisibility sktime time series distances.

    Parameters
    ----------
    distance : str, or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a :term:`Glossary <sparse graph>`, in which
        case only "nonzero" elements may be considered neighbors for DBSCAN.
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important Kmeans parameter to choose appropriately for your data set
        and distance function.
    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
    leaf_size : int, default=30
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.
    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "seigpe",
        # estimator type
        # --------------
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "X_inner_mtype": ["pd-multiindex", "numpy3D"],
        # required by the update_data utility
        # otherwise, we could pass through to the distance directly
        "capability:out_of_sample": False,
        "capability:predict": True,
        "capability:predict_proba": False,
    }

    DELEGATED_PARAMS = ["eps", "min_samples", "algorithm", "leaf_size", "n_jobs"]
    DELEGATED_FITTED_PARAMS = ["core_sample_indices_", "components_ ", "labels_"]

    def __init__(
        self,
        distance,
        eps=0.5,
        min_samples=5,
        algorithm="auto",
        leaf_size=30,
        n_jobs=None,
        n_init=4
    ):
        self.distance = distance
        self.eps = eps
        self.min_samples = min_samples
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs,
        self.n_init = n_init

        super().__init__()

        if isinstance(distance, BasePairwiseTransformerPanel):
            tags_to_clone = [
                "capability:unequal_length",
                "capability:missing_values",
            ]
            self.clone_tags(distance, tags_to_clone)

        # numba distance in sktime (indexed by string)
        # cannot support unequal length data, and require numpy3D input
        if isinstance(distance, str):
            tags_to_set = {
                "X_inner_mtype": "numpy3D",
                "capability:unequal_length": False,
            }
            self.set_tags(**tags_to_set)

        self.kmeans_ = None

    def _ts_to_graph(self, X):
        ts_attr = []
        X_ts = []
        print(X.shape)

        for i in range(len(X)):
            X_ts.append(X[i].reshape(1, X[1].shape[0])[0])
        for ts in X_ts:
            # ts for each time series
            g = HorizontalVG()
            g.build(ts)
            nx_g = g.as_networkx()

            density_h = nx.density(nx_g)
            max_grade_h = max(nx_g.degree, key=lambda x: x[1])[1]

            # Natural VG
            gn = NaturalVG()
            gn.build(ts)
            nx_gn = gn.as_networkx()
            density_n = nx.density(nx_gn)
            max_grade_n = max(nx_gn.degree, key=lambda x: x[1])[1]

            ts_attr.append([density_h, max_grade_h, density_n, max_grade_n])
        df = pd.DataFrame(
            ts_attr, columns=["density_h", "max_degree_h",
                              "density_n", "max_degree_n"]
        )

        ts_features = np.array(
            df[["density_h", "max_degree_h", "density_n", "max_degree_n"]]
        )
        return ts_features

    def _fit(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : panel of time series, any sklearn Panel mtype
            Time series to fit clusters to
        y: ignored, exists for API consistency reasons

        Returns
        -------
        self:
            Fitted estimator.
        """
        self._X = X

        deleg_param_dict = {key: getattr(self, key) for key in self.DELEGATED_PARAMS}
        
        self.kmeans_ = None

        self.ts_features = self._ts_to_graph(X)

        self.kmeans_ = KMeans(init="k-means++", n_clusters=self.n_clusters, n_init=4, **deleg_param_dict)
        self.kmeans_.fit(self.ts_features)

        for key in self.DELEGATED_FITTED_PARAMS:
            if hasattr(self.kmeans_, key):
                setattr(self, key, getattr(self.kmeans_, key))
        return self

    def _fit_predict(self, X, y=None):
        """Fit time series clusterer to training data.

        Parameters
        ----------
        X : panel of time series, any sklearn Panel mtype
            Time series to fit clusters to
        y: ignored, exists for API consistency reasons

        Returns
        -------
        self:
             labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        self._X = X



        deleg_param_dict = {key: getattr(self, key) for key in self.DELEGATED_PARAMS}
        
        self.kmeans_ = None

        self.ts_features = self._ts_to_graph(X)

        self.kmeans_ = KMeans(init="k-means++", n_clusters=self.n_clusters, n_init=4, **deleg_param_dict)
        self.kmeans_.fit(self.ts_features)
        
        for key in self.DELEGATED_FITTED_PARAMS:
            if hasattr(self.kmeans_, key):
                setattr(self, key, getattr(self.kmeans_, key))

        return self.kmeans_.predict(self.ts_features)

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
            self.ts_features = self._ts_to_graph(X)
            return self.clone().fit_predict(self.ts_features)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.dists_kernels import AggrDist, DtwDist, EditDist

        params1 = {"distance": DtwDist()}
        params2 = {"distance": EditDist()}

        # distance capable of unequal length
        dist = AggrDist.create_test_instance()
        params3 = {"distance": dist}

        return [params1, params2, params3]
