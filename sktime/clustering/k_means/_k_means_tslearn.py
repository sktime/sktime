"""Time series kernel kmeans."""

import numpy as np

from sktime.base.adapters._tslearn import _TslearnAdapter
from sktime.clustering.base import BaseClusterer


class TimeSeriesKMeansTslearn(_TslearnAdapter, BaseClusterer):
    """K-means clustering for time-series data, from tslearn.

    Direct interface to ``tslearn.clustering.TimeSeriesKMeans``.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than
        this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm
        stops.

    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia.

    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric to be used for both cluster assignment and barycenter
        computation. If "dtw", DBA is used for barycenter
        computation.

    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process. Only used
        if ``metric="dtw"`` or ``metric="softdtw"``.

    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
        For metrics that accept parallelization of the cross-distance matrix
        computations, ``n_jobs`` key passed in ``metric_params`` is overridden by
        the ``n_jobs`` argument.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    dtw_inertia: bool (default: False)
        Whether to compute DTW inertia even if DTW is not the chosen metric.

    verbose : int (default: 0)
        If nonzero, print information about the inertia while learning
        the model and joblib progress messages are printed.

    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it
        fixes the seed. Defaults to the global
        numpy random number generator.

    init : {'k-means++', 'random' or an ndarray} (default: 'random')
        Method for initialization:
        'k-means++' : use k-means++ heuristic. See `scikit-learn's k_init_
        <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/\
        cluster/k_means_.py>`_ for more.
        'random': choose k observations (rows) at random from data for the
        initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, ts_size, d)
        and gives the initial centers.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point.

    cluster_centers_ : numpy.ndarray of shape (n_clusters, sz, d)
        Cluster centers.
        ``sz`` is the size of the time series used at fit time if the init method
        is 'k-means++' or 'random', and the size of the longest initial
        centroid if those are provided as a numpy array through init parameter.

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    n_iter_ : int
        The number of iterations performed during fit.

    Notes
    -----
        If ``metric`` is set to ``"euclidean"``, the algorithm expects a dataset of
        equal-sized time series.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["rtavenar", "fkiraly"],  # rtavenar credit for interfaced code
        "python_dependencies": "tslearn",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": False,
    }

    # defines the name of the attribute containing the tslearn estimator
    _estimator_attr = "_tslearn_k_means"

    def _get_tslearn_class(self):
        """Get tslearn class.

        should import and return tslearn class
        """
        from tslearn.clustering import TimeSeriesKMeans as TsLearnKMeans

        return TsLearnKMeans

    def __init__(
        self,
        n_clusters=3,
        max_iter=50,
        tol=1e-6,
        n_init=1,
        metric="euclidean",
        max_iter_barycenter=100,
        metric_params=None,
        n_jobs=None,
        dtw_inertia=False,
        verbose=0,
        random_state=None,
        init="random",
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.metric = metric
        self.max_iter_barycenter = max_iter_barycenter
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.dtw_inertia = dtw_inertia
        self.verbose = verbose
        self.random_state = random_state
        self.init = init

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        super().__init__(n_clusters=n_clusters)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {
            "n_clusters": 3,
            "max_iter": 3,
            "tol": 0.001,
            "n_init": 2,
            "metric": "euclidean",
            "max_iter_barycenter": 7,
            "verbose": 0,
            "random_state": 42,
            "init": "k-means++",
        }
        params2 = {
            "n_clusters": 2,
            "max_iter": 5,
            "tol": 0.0001,
            "n_init": 1,
            "metric": "dtw",
            "max_iter_barycenter": 10,
            "verbose": 0,
            "random_state": None,
            "init": "random",
        }
        return [params1, params2]

    def _score(self, X, y=None) -> float:
        return np.abs(self.inertia_)
