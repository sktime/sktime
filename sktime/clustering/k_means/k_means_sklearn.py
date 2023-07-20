"""Time series KMeans, wrapping sklearn KMeans."""

import numpy as np
from sklearn.cluster import KMeans

from sktime.clustering.base import BaseClusterer
from sktime.dists_kernels._base import BasePairwiseTransformerPanel


class TimeSeriesKMeansSklearn(BaseClusterer):
    """KMeans for time series distances.

    Interface to sklearn KMeans with sktime time series distances.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.
    n_init : 'auto' or int, default='auto'
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'` or `init` is a callable;
        1 if using `init='k-means++'` or `init` is an array-like.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose : int, default=0
        Verbosity mode.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.
    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "X_inner_mtype": ["pd-multiindex", "numpy3D"],
        # required by the update_data utility
        # otherwise, we could pass through to the distance directly
    }

    DELEGATED_PARAMS = [
        "n_clusters",
        "init",
        "n_init",
        "max_iter",
        "tol",
        "verbose",
        "random_state",
        "copy_x",
        "algorithm",
    ]
    DELEGATED_FITTED_PARAMS = ["cluster_centers_", "labels_"]

    def __init__(
        self,
        distance,
        n_clusters=8,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        self.distance = distance
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

        super().__init__(n_clusters)

        if isinstance(distance, BasePairwiseTransformerPanel):
            tags_to_clone = [
                "capability:multivariate",
                "capability:unequal_length",
                "capability:missing_values",
            ]
            self.clone_tags(distance, tags_to_clone)

        self.kmeans_ = None

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

        distance = self.distance
        distmat = distance(X)

        deleg_param_dict = {key: getattr(self, key) for key in self.DELEGATED_PARAMS}

        self.kmeans_ = KMeans(**deleg_param_dict)
        self.kmeans_.fit(X=distmat)

        for key in self.DELEGATED_FITTED_PARAMS:
            if hasattr(self.kmeans_, key):
                setattr(self, key, getattr(self.kmeans_, key))

        return self

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
        distance = self.distance
        distmat = distance(X)

        return self.kmeans_.predict(distmat)

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        Default behaviour is to call _predict and set the predicted class probability
        to 1, other class probabilities to 0. Override if better estimates are
        obtainable.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        preds = self._predict(X)
        n_instances = len(preds)
        n_clusters = max(preds) + 1
        dists = np.zeros((X.shape[0], n_clusters))
        for i in range(n_instances):
            dists[i, preds[i]] = 1
        return dists

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
