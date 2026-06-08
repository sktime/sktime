"""Time series Kvisibility."""

__author__ = ["seigpe"]

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sktime.clustering.base import BaseClusterer


class TimeSeriesKvisibility(BaseClusterer):
    """Kvisibility for time series clustering.

    kvisibility is a time series clustering technique based on visibility graphs.
    The algorithm is based on the transformation of the time series into graphs,
    and with metrics of the created graphs create a clustering with Kmeans.

    Based on the paper [1]_.

    Parameters
    ----------
    init : {'k-means++', 'random'}, callable or
        array-like of shape (n_clusters, n_features), default='k-means++'
        Method for initialization:
        'k-means++' : selects initial cluster centroids using sampling based
        on an empirical probability distribution of the points' contribution
        to the overall inertia. This technique speeds up convergence. The
        algorithm implemented is “greedy k-means++”. It differs from the
        vanilla k-means++ by making several trials at each sampling step
        and choosing the best centroid among them.
        'random': choose n_clusters observations (rows) at random from data
        for the initial centroids.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.
    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid seeds.
        The final results is the best output of n_init consecutive runs in terms
        of inertia. Several runs are recommended for sparse high-dimensional problems.
        When n_init='auto', the number of runs depends on the value of init: 10 if
        using init='random' or init is a callable; 1 if using init='k-means++' or
        init is an array-like.
    n_clusters : int, default=5
        The number of clusters to form as well as the number of centroids to generate.

    Attributes
    ----------
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    References
    ----------
    .. [1]  https://www.aimspress.com/article/doi/10.3934/math.20241687
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "seigpe",
        "maintainers": ["seigpe", "acoxonante"],
        "python_dependencies": ["networkx", "ts2vg"],
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

    DELEGATED_PARAMS = ["init", "n_clusters", "n_init"]
    DELEGATED_FITTED_PARAMS = ["core_sample_indices_", "components_ ", "labels_"]

    def __init__(self, n_clusters=5, init="k-means++", n_init=4):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init

        super().__init__(n_clusters=n_clusters)

        self.kmeans_ = None

    def _ts_to_graph(self, X):
        import networkx as nx
        from ts2vg import HorizontalVG, NaturalVG

        ts_attr = []
        X_ts = []

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
            ts_attr, columns=["density_h", "max_degree_h", "density_n", "max_degree_n"]
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

        self.kmeans_ = KMeans(
            init=self.init,
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            **deleg_param_dict,
        )
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

        self.kmeans_ = KMeans(
            init=self.init,
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            **deleg_param_dict,
        )
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
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid
            test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params0 = {}
        params1 = {"n_clusters": 5}
        params2 = {"n_init": 4}
        params3 = {"n_clusters": 4, "n_init": 6}

        return [params0, params1, params2, params3]
