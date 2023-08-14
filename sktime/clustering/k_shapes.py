"""Time series kshapes."""
from typing import Union

import numpy as np
from numpy.random import RandomState

from sktime.base.adapters._tslearn import _TslearnAdapter
from sktime.clustering.base import BaseClusterer


class TimeSeriesKShapes(_TslearnAdapter, BaseClusterer):
    """Kshape clustering for time series.

    Direct interface to ``tslearn.clustering.KShape``.

    Parameters
    ----------
    n_clusters: int, defaults = 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init_algorithm: str or np.ndarray, defaults = 'random'
        Method for initializing cluster centers. Any of the following are valid:
        ['random']. Or a np.ndarray of shape (n_clusters, ts_size, d) and gives the
        initial centers.
    n_init: int, defaults = 10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final result will be the best output of n_init
        consecutive runs in terms of inertia.
    max_iter: int, defaults = 30
        Maximum number of iterations of the k-means algorithm for a single
        run.
    tol: float, defaults = 1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
    verbose: bool, defaults = False
        Verbosity mode.
    random_state: int or np.random.RandomState instance or None, defaults = None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    labels_: np.ndarray (1d array of shape (n_instance,))
        Labels that is the index each time series belongs to.
    inertia_: float
        Sum of squared distances of samples to their closest cluster center, weighted by
        the sample weights if provided.
    n_iter_: int
        Number of iterations run.
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "python_dependencies": "tslearn",
    }

    # defines the name of the attribute containing the tslearn estimator
    _estimator_attr = "_tslearn_k_shapes"

    def __init__(
        self,
        n_clusters: int = 8,
        init_algorithm: Union[str, np.ndarray] = "random",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = False,
        random_state: Union[int, RandomState] = None,
    ):
        self.init_algorithm = init_algorithm
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        super().__init__(n_clusters=n_clusters)

    def _get_tslearn_class(self):
        """Get tslearn class.

        should import and return tslearn class
        """
        from tslearn.clustering import KShape

        return KShape

    def _get_tslearn_object(self):
        """Initialize tslearn object.

        We need to override this due to the different names of
        init_algorithm, which in tslearn is init
        """
        cls = self._get_tslearn_class()
        params = self.get_params()
        params["init"] = params.pop("init_algorithm")
        return cls(**params)

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
        params1 = {
            "n_clusters": 3,
            "n_init": 2,
            "max_iter": 2,
            "tol": 1e-3,
            "verbose": False,
            "random_state": 2,
        }
        params2 = {
            "n_clusters": 2,
            "init_algorithm": "random",
            "n_init": 1,
            "max_iter": 1,
            "tol": 1e-4,
            "verbose": False,
            "random_state": 1,
        }
        return [params1, params2]

    def _score(self, X, y=None):
        return np.abs(self.inertia_)
