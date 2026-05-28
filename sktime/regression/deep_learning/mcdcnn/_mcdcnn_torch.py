"""Multi Channel Deep Convolutional Neural Regressor (MCDCNN)."""

from sktime.networks.mcdcnn import MCDCNNNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class MCDCNNRegressorTorch(BaseDeepRegressorTorch):
    """Multi Channel Deep Convolutional Neural Regressor in PyTorch, adopted from [1]_.

    Adapted from the implementation of Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mcdcnn.py

    Parameters
    ----------
    n_epochs : int, optional (default=120)
        The number of epochs to train the model.
    batch_size : int, optional (default=16)
        The number of samples per gradient update.
    kernel_sizes : tuple, optional (default=(5, 5))
        The size of kernels in Conv1D layers.
    pool_size : int, optional (default=2)
        The size of kernel in (Max) Pool layer.
    filter_sizes : tuple, optional (default=(8, 8))
        The sizes of filter for Conv1D layer corresponding
        to each Conv1D in the block.
    dense_units : int, optional (default=732)
        The number of output units of the final Dense
        layer of this Network. This is NOT the final layer
        but the penultimate layer.
    conv_padding : str or None, optional (default="same")
        The type of padding to be applied to convolutional
        layers.
    pool_padding : str or None, optional (default="same")
        The type of padding to be applied to pooling layers.
    criterion : str, optional (default="MSELoss")
        The name of the loss function to be used during training,
        should be supported by PyTorch.
    criterion_kwargs : dict or None, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    activation : str or None, optional (default=None)
        The activation function to apply at the output.
        List of available activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-activation
    activation_hidden : string, default="relu"
        Activation function used in the hidden layers.
        List of available activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-activation
    use_bias : bool, optional (default=True)
        Whether bias should be included in the output layer.
    optim: str or None or an instance of optimizers defined in torch.optim,
        optional (default=None)
        The optimizer to use for training the model. If left with None, "SGD" is
        used with momentum=0.9, weight_decay=0.0005.
        List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms

    optim_kwargs : dict or None, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
        If None, SGD is used with momentum=0.9, weight_decay=0.0005.
    callbacks : None or str or a tuple of str, optional (default=None)
        Currently only learning rate schedulers are supported as callbacks.
        If more than one scheduler is passed, they are applied sequentially in the
        order they are passed. If None, then no learning rate scheduler is used.
    callback_kwargs : dict or None, optional (default=None)
        The keyword arguments to be passed to the callbacks.
    lr : float, optional (default=0.01)
        The learning rate to use for the optimizer.
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    random_state : int, optional (default=0)
        The seed to any random action.

    References
    ----------
    .. [1] Zheng et. al, Time series classification using multi-channels deep
      convolutional neural networks, International Conference on
      Web-Age Information Management, Pages 298-310, year 2014, organization: Springer.

    Examples
    --------
    >>> from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressorTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> mcdcnn = MCDCNNRegressorTorch(n_epochs=1, kernel_sizes=(4, 4)) # doctest: +SKIP
    >>> mcdcnn.fit(X_train, y_train) # doctest: +SKIP
    MCDCNNRegressorTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["hfawaz", "James-Large", "noxthot"],
        "python_dependencies": "torch",
        # estimator type handled by parent class
    }

    def __init__(
        self: "MCDCNNRegressorTorch",
        n_epochs=120,
        batch_size=16,
        kernel_sizes=(5, 5),
        pool_size=2,
        filter_sizes=(8, 8),
        dense_units=732,
        conv_padding="same",
        pool_padding="same",
        criterion="MSELoss",
        criterion_kwargs=None,
        activation=None,
        activation_hidden="relu",
        use_bias=True,
        optim=None,
        optim_kwargs=None,
        callbacks=None,
        callback_kwargs=None,
        lr=0.01,
        verbose=False,
        random_state=0,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        self.pool_size = pool_size
        self.filter_sizes = filter_sizes
        self.dense_units = dense_units
        self.conv_padding = conv_padding
        self.pool_padding = pool_padding
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.use_bias = use_bias

        # stored as-is: sklearn __init__ contract requires no mutation
        self.optim = optim
        self.optim_kwargs = optim_kwargs

        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # resolve defaults as local vars so the parent stores conformant values
        _optimizer = optim if optim is not None else "SGD"
        _optimizer_kwargs = (
            optim_kwargs
            if not (optim is None and optim_kwargs is None)
            else {"momentum": 0.9, "weight_decay": 0.0005}
        )

        super().__init__(
            num_epochs=self.n_epochs,
            batch_size=self.batch_size,
            criterion=self.criterion,
            criterion_kwargs=self.criterion_kwargs,
            optimizer=_optimizer,
            optimizer_kwargs=_optimizer_kwargs,
            callbacks=self.callbacks,
            callback_kwargs=self.callback_kwargs,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * dynamic tag setting
        * any soft dependency imports in the constructor
        """
        if len(self.filter_sizes) != len(self.kernel_sizes):
            raise ValueError(
                f"Length of `filter_sizes` {len(self.filter_sizes)} must match "
                f"the number of convolutional layers determined by the length of tuple "
                f"`kernel_sizes` {len(self.kernel_sizes)}."
            )

        super().__post_init__()

    def _build_network(self, X):
        """Build the MCDCNN network with output layer for regression.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.

        Returns
        -------
        model : torch.nn.Module
            The constructed MCDCNN network with output layer.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )

        return MCDCNNNetworkTorch(
            kernel_sizes=self.kernel_sizes,
            pool_size=self.pool_size,
            filter_sizes=self.filter_sizes,
            dense_units=self.dense_units,
            conv_padding=self.conv_padding,
            pool_padding=self.pool_padding,
            random_state=self.random_state,
            activation=self.activation,
            activation_hidden=self.activation_hidden,
            use_bias=self.use_bias,
        )

    def _instantiate_optimizer(self):
        """Instantiate optimizer from the sklearn-conformant optim/optim_kwargs params.

        Uses self.optim and self.optim_kwargs (the params declared in __init__) rather
        than self.optimizer/self.optimizer_kwargs (set by the parent) so that
        set_params(optim=...) propagates correctly to the optimizer used in training.
        """
        from sktime.utils.dependencies import _safe_import

        params = self.network.parameters()
        if self.optim is None:
            kwargs = (
                self.optim_kwargs
                if self.optim_kwargs is not None
                else {"momentum": 0.9, "weight_decay": 0.0005}
            )
            SGD = _safe_import("torch.optim.SGD")
            return SGD(params, lr=self.lr, **kwargs)
        if self._all_optimizers is None:
            self._all_optimizers = {
                "adadelta": "Adadelta",
                "adagrad": "Adagrad",
                "adam": "Adam",
                "adamw": "AdamW",
                "sparseadam": "SparseAdam",
                "adamax": "Adamax",
                "asgd": "ASGD",
                "lbfgs": "LBFGS",
                "nadam": "NAdam",
                "radam": "RAdam",
                "rmsprop": "RMSprop",
                "rprop": "Rprop",
                "sgd": "SGD",
            }
        torchOptimizer = _safe_import("torch.optim.Optimizer")
        if isinstance(self.optim, str):
            key = self.optim.lower()
            if key not in self._all_optimizers:
                raise ValueError(
                    f"Unknown optimizer: '{self.optim}'. Please pass one of "
                    f"{', '.join(self._all_optimizers)} for `optim`."
                )
            optimizer_class = _safe_import(
                f"torch.optim.{self._all_optimizers[key]}"
            )
            kwargs = self.optim_kwargs if self.optim_kwargs is not None else {}
            return optimizer_class(params, lr=self.lr, **kwargs)
        if isinstance(self.optim, torchOptimizer):
            return self.optim
        raise TypeError(
            "`optim` must be None, a str, or a torch.optim.Optimizer instance. "
            f"Got {type(self.optim)} instead."
        )

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
        # default: optim=None exercises the default-SGD path
        params1 = {
            "n_epochs": 1,
            "random_state": 0,
        }
        params2 = {
            "n_epochs": 1,
            "batch_size": 16,
            "kernel_sizes": (5,),
            "pool_size": 2,
            "filter_sizes": (8,),
            "dense_units": 10,
            "use_bias": True,
            "activation": None,
            "random_state": 0,
        }
        # explicit Adam + kwargs: exercises set_params propagation
        params3 = {
            "n_epochs": 2,
            "batch_size": 2,
            "kernel_sizes": (7, 7),
            "pool_size": 2,
            "filter_sizes": (8, 8),
            "dense_units": 1,
            "conv_padding": "same",
            "pool_padding": "same",
            "activation_hidden": "relu",
            "use_bias": False,
            "optim": "Adam",
            "optim_kwargs": {"weight_decay": 0.001},
            "lr": 0.01,
            "random_state": 0,
        }
        # optim=None, explicit optim_kwargs: custom SGD kwargs stored as-is
        params4 = {
            "n_epochs": 1,
            "kernel_sizes": (5,),
            "filter_sizes": (8,),
            "optim": None,
            "optim_kwargs": {"momentum": 0.5, "weight_decay": 0.001},
            "random_state": 0,
        }

        return [params1, params2, params3, params4]
