"""Time series k-neighbors, from tslearn."""

from sktime.base.adapters._tslearn import _TslearnAdapter
from sktime.classification.base import BaseClassifier


class KNeighborsTimeSeriesClassifierTslearn(_TslearnAdapter, BaseClassifier):
    """K-nearest neighbors Time Series Classifier, from tslearn.

    Direct interface to ``tslearn.neighbors.KNeighborsTimeSeriesClassifier``.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.

    weights : str or callable, optional (default: 'uniform')
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this
          case, closer neighbors of a query point
          will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of
          distances, and returns an array of the same
          shape containing the weights.

    metric : {'dtw', 'softdtw', 'ctw', 'euclidean', 'sqeuclidean', 'cityblock',  'sax'}
        default: 'dtw'.
        Metric to be used at the core of the nearest neighbor procedure.
        When ``'sax'`` is provided as a metric, the data is expected to be
        normalized such that each time series has zero mean and unit
        variance. ``'euclidean'``, ``'sqeuclidean'``, ``'cityblock'`` are described
        in `scipy.spatial.distance doc
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_.

    metric_params : dict or None (default: None)
        Dictionary of metric parameters.
        For metrics that accept parallelization of the cross-distance matrix
        computations, ``n_jobs`` and ``verbose`` keys passed in ``metric_params``
        are overridden by the ``n_jobs`` and ``verbose`` arguments.
        For ``'sax'`` metric, these are hyper-parameters to be passed at the
        creation of the ``SymbolicAggregateApproximation`` object.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    Examples
    --------
    >>> import sktime.classification.distance_based as db_clf  # doctest: +SKIP
    >>> from db_clf import KNeighborsTimeSeriesClassifierTslearn  # doctest: +SKIP
    >>> from sktime.datasets import load_unit_test  # doctest: +SKIP
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test")  # doctest: +SKIP
    >>> clf = KNeighborsTimeSeriesClassifierTslearn(
    ...     n_neighbors=5,
    ...     weights="uniform",
    ...     metric="dtw",
    ...     metric_params=None,
    ...     n_jobs=None,
    ...     verbose=0,
    ... )  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    KNeighborsTimeSeriesClassifierTslearn(...)
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
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
    _estimator_attr = "_tslearn_knn"

    def _get_tslearn_class(self):
        """Get tslearn class.

        should import and return tslearn class
        """
        from tslearn.neighbors import KNeighborsTimeSeriesClassifier as TsLearnKnn

        return TsLearnKnn

    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        metric="dtw",
        metric_params=None,
        n_jobs=None,
        verbose=0,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose

        super().__init__()

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
        params1 = {}
        params2 = {
            "n_neighbors": 3,
            "weights": "distance",
            "metric": "sax",
            "metric_params": None,
        }
        return [params1, params2]
