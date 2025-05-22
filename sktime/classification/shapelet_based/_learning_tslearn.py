"""Learning Shapelets Classifier, from tslearn."""

from sktime.base.adapters._tslearn import _TslearnAdapter
from sktime.classification.base import BaseClassifier


class ShapeletLearningClassifierTslearn(_TslearnAdapter, BaseClassifier):
    """Learning Time Series Shapelets Classifier, from tslearn.

    Direct interface to ``tslearn.shapelets.shapelets.LearningShapelets``.

    Learning Time-Series Shapelets was originally presented in [1]_.

    Parameters
    ----------
    n_shapelets_per_size: dict (default: None)
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value).
        If None, `grabocka_params_to_shapelet_size_dict` is used and the
        size used to compute is that of the shortest time series passed at fit time.

    max_iter: int (default: 10,000)
        Number of training epochs.

    batch_size: int (default: 256)
        Batch size to be used.

    optimizer: str or keras.optimizers.Optimizer (default: "sgd")
        ``keras`` optimizer to use for training.

    weight_regularizer: float or None (default: 0.)
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer. If 0, no regularization is performed.

    shapelet_length: float (default: 0.15)
        The length of the shapelets, expressed as a fraction of the time series length.
        Used only if ``n_shapelets_per_size`` is None.

    total_lengths: int (default: 3)
        The number of different shapelet lengths. Will extract shapelets of
        length i * shapelet_length for i in [1, total_lengths]
        Used only if ``n_shapelets_per_size`` is None.

    max_size: int or None (default: None)
        Maximum size for time series to be fed to the model. If None, it is
        set to the size (number of timestamps) of the training time series.

    scale: bool (default: False)
        Whether input data should be scaled for each feature of each time
        series to lie in the [0-1] interval.
        Default for this parameter is set to `False` in version 0.4 to ensure
        backward compatibility, but is likely to change in a future version.

    verbose: {0, 1, 2} (default: 0)
        ``keras`` verbose level.

    random_state : int or None, optional (default: None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, ``random_state`` is the seed used by the random number
        generator; If None, the random number generator is the ``RandomState``
        instance used by ``np.random``.

    Attributes
    ----------
    shapelets_ : numpy.ndarray of objects, each object being a time series
        Set of time-series shapelets.

    shapelets_as_time_series_ : numpy.ndarray of shape (n_shapelets, sz_shp, d)
        where ``sz_shp`` is the maximum of all shapelet sizes
        Set of time-series shapelets formatted as a ``tslearn`` time series dataset.

    transformer_model_ : keras.Model
        Transforms an input dataset of timeseries into distances to the
        learned shapelets.

    locator_model_ : keras.Model
        Returns the indices where each of the shapelets can be found (minimal
        distance) within each of the timeseries of the input dataset.

    model_ : keras.Model
        Directly predicts the class probabilities for the input timeseries.

    history_ : dict
        Dictionary of losses and metrics recorded during fit.

    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["rtavenar", "yanncabanes", "fspinna", "fkiraly"],
        "python_dependencies": "tslearn",
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": False,
    }

    # defines the name of the attribute containing the tslearn estimator
    _estimator_attr = "_tslearn_shapelets"

    def _get_tslearn_class(self):
        """Get tslearn class.

        should import and return tslearn class
        """
        from tslearn.shapelets.shapelets import LearningShapelets

        return LearningShapelets

    def __init__(
        self,
        n_shapelets_per_size=None,
        max_iter=10000,
        batch_size=256,
        optimizer="sgd",
        weight_regularizer=0.0,
        shapelet_length=0.15,
        total_lengths=3,
        max_size=None,
        scale=False,
        verbose=0,
        random_state=None,
    ):
        self.n_shapelets_per_size = n_shapelets_per_size
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.optimizer = optimizer
        self.weight_regularizer = weight_regularizer
        self.shapelet_length = shapelet_length
        self.total_lengths = total_lengths
        self.max_size = max_size
        self.scale = scale
        self.random_state = random_state

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
        params1 = {"max_iter": 120, "batch_size": 64}
        params2 = {
            "max_iter": 100,
            "batch_size": 128,
            "optimizer": "adam",
            "shapelet_length": 0.1,
            "scale": True,
        }
        return [params1, params2]
