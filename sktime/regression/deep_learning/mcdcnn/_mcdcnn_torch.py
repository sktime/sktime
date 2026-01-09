"""Multi Channel Deep Convolutional Neural Regressor (MCDCNN)."""


from sktime.networks.mcdcnn import MCDCNNNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch
from sktime.utils.dependencies import _check_dl_dependencies


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
    kernel_size : int, optional (default=5)
        The size of kernel in Conv1D layer.
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
    optimizer : str or None or an instance of optimizers defined in torch.optim,
        optional (default=None)
        The optimizer to use for training the model. If None, SGD is used with
        learning_rate=0.01, momentum=0.9, weight_decay=0.0005.
        List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict or None, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
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
    >>> mcdcnn = MCDCNNRegressorTorch(n_epochs=1, kernel_size=4) # doctest: +SKIP
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
        kernel_size=5,
        pool_size=2,
        filter_sizes=(8, 8),
        dense_units=732,
        conv_padding="same",
        pool_padding="same",
        criterion="MSELoss",
        activation=None,
        activation_hidden="relu",
        use_bias=True,
        optimizer=None,
        optimizer_kwargs=None,
        callbacks=None,
        callback_kwargs=None,
        lr=0.01,
        verbose=False,
        random_state=0,
    ):
        _check_dl_dependencies(severity="error")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.filter_sizes = filter_sizes
        self.dense_units = dense_units
        self.conv_padding = conv_padding
        self.pool_padding = pool_padding
        self.criterion = criterion
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # Set default optimizer to SGD with TF version defaults if not provided
        if self.optimizer is None:
            self.optimizer = "SGD"
            if self.optimizer_kwargs is None:
                self.optimizer_kwargs = {"momentum": 0.9, "weight_decay": 0.0005}

        super().__init__(
            num_epochs=self.n_epochs,
            batch_size=self.batch_size,
            criterion=self.criterion,
            criterion_kwargs=None,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            callbacks=self.callbacks,
            callback_kwargs=self.callback_kwargs,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

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
            kernel_size=self.kernel_size,
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
            "n_epochs": 1,
            "batch_size": 2,
            "kernel_size": 4,
            "pool_size": 2,
            "filter_sizes": (4, 4),
            "dense_units": 10,
            "use_bias": False,
            "activation": None,
        }
        return [params1, params2]
