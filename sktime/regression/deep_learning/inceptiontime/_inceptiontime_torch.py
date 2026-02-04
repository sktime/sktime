"""InceptionTime Regressor in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["InceptionTimeRegressorTorch"]

from collections.abc import Callable

from sktime.networks.inceptiontime import InceptionTimeNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class InceptionTimeRegressorTorch(BaseDeepRegressorTorch):
    """InceptionTime Deep Learning Regressor in PyTorch.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    Parameters
    ----------
    num_epochs : int, default=1500
        The number of epochs to train the model.
    batch_size : int, default=64
        The size of each mini-batch during training.
    kernel_size : int, default=40
        Base kernel size for inception modules
    n_filters : int, default=32
        Number of filters in the convolution layers
    use_residual : bool, default=True
        If True, uses residual connections
    use_bottleneck : bool, default=True
        If True, uses bottleneck layer in inception modules
    bottleneck_size : int, default=32
        Size of the bottleneck layer
    depth : int, default=6
        Number of inception modules to stack
    activation : str or None, default=None
        Activation function used for the final output layer.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu', None
    activation_hidden : str, default="relu"
        Activation function used for hidden layers (output from inception modules).
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    activation_inception : str or None, default=None
        Activation function used inside the inception modules.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu', None
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "Adam"
        The optimizer to use for training the model.
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    criterion : case insensitive str or None or an instance of a loss function
        defined in PyTorch, default = "MSELoss"
        The loss function to be used in training the neural network.
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the loss function.
    callbacks : None or str or a tuple of str, default = None
        Currently only learning rate schedulers are supported as callbacks.
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.
    random_state : int or None, default = None
        Seed to ensure reproducibility.

    Notes
    -----
    ..[1] Fawaz et. al, InceptionTime: Finding AlexNet for Time Series
    Classification, Data Mining and Knowledge Discovery, 34, 2020

    Examples
    --------
    >>> from sktime.regression.deep_learning.inceptiontime import (
    ...     InceptionTimeRegressorTorch
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> reg = InceptionTimeRegressorTorch(num_epochs=50, batch_size=2)  # doctest: +SKIP
    >>> reg.fit(X_train, y_train)  # doctest: +SKIP
    InceptionTimeRegressorTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Faakhir30"],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "InceptionTimeRegressorTorch",
        # model specific
        n_filters: int = 32,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
        depth: int = 6,
        kernel_size: int = 40,
        activation: str | None = None,
        activation_hidden: str = "relu",
        activation_inception: str = "linear",
        # base regressor specific
        num_epochs: int = 1500,
        batch_size: int = 64,
        optimizer: str | None | Callable = "Adam",
        criterion: str | None | Callable = "MSELoss",
        callbacks: None | str | tuple[str, ...] = None,
        criterion_kwargs: dict | None = None,
        optimizer_kwargs: dict | None = None,
        callback_kwargs: dict | None = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.bottleneck_size = bottleneck_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.activation_inception = activation_inception
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # input_size to be inferred from the data
        # and will be set in _build_network
        self.input_size = None
        self.num_classes = 1  # because regression

        super().__init__(
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            criterion=self.criterion,
            criterion_kwargs=self.criterion_kwargs,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            callbacks=self.callbacks,
            callback_kwargs=self.callback_kwargs,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _build_network(self, X):
        """Build the InceptionTime network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.

        Returns
        -------
        model : InceptionTimeNetworkTorch instance
            The constructed InceptionTime network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )
        # n_instances, n_dims, n_timesteps = X.shape
        X = X.transpose(0, 2, 1)  # to (n_instances, n_timesteps, n_dims)
        _, self.input_size, _ = X.shape
        return InceptionTimeNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            n_filters=self.n_filters,
            use_residual=self.use_residual,
            use_bottleneck=self.use_bottleneck,
            bottleneck_size=self.bottleneck_size,
            depth=self.depth,
            kernel_size=self.kernel_size,
            random_state=self.random_state,
            activation=self.activation,
            activation_hidden=self.activation_hidden,
            activation_inception=self.activation_inception,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """
        params1 = {
            "n_filters": 8,
            "depth": 3,
            "num_epochs": 10,
            "batch_size": 2,
        }
        params2 = {
            "n_filters": 16,
            "use_residual": True,
            "use_bottleneck": True,
            "depth": 6,
            "kernel_size": 20,
            "num_epochs": 12,
            "batch_size": 4,
        }

        return [params1, params2]
