"""InceptionTime Classifier in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["InceptionTimeClassifierTorch"]

import warnings
from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.inceptiontime import InceptionTimeNetworkTorch


class InceptionTimeClassifierTorch(BaseDeepClassifierPytorch):
    """InceptionTime Deep Learning Classifier in PyTorch.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py

    InceptionTimeClassifierTorch is a single instance of InceptionTime model
    described in the original publication [1]_, which uses an ensemble of 5
    single instances.

    To build an ensemble of models mirroring [1]_, use the BaggingClassifier with
    n_estimators=5, bootstrap=False, and estimator being an instance of
    this InceptionTimeClassifierTorch.

    Parameters
    ----------
    num_epochs : int, default=1500
        The number of epochs to train the model.
    n_filters : int, default=32
        Number of filters in the convolution layers
    batch_size : int, default=64
        The size of each mini-batch during training.
    kernel_size : int, default=40
        Base kernel size for inception modules
    use_residual : bool, default=True
        If True, uses residual connections
    use_bottleneck : bool, default=True
        If True, uses bottleneck layer in inception modules.
    bottleneck_size : int, default=32
        Size of the bottleneck layer.
    depth : int, default=6
        Number of inception modules to stack.
    activation: str or None, default=None
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
        defined in PyTorch, default = "CrossEntropyLoss"
        The loss function to be used in training the neural network.
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the loss function.
    callbacks : None or str or a tuple of str, default = None
        Currently only learning rate schedulers are supported as callbacks.
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    weights_init : str or None, default = None
        The method to initialize the weights of the conv layers. Supported values are
        'kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', or None
        for default PyTorch initialization.
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
    Single instance of InceptionTime model:
    >>> from sktime.classification.deep_learning.inceptiontime import (
    ...     InceptionTimeClassifierTorch
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = InceptionTimeClassifierTorch( # doctest: +SKIP
    ...     num_epochs=50, batch_size=2
    ... )
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    InceptionTimeClassifierTorch(...)

    To build an ensemble of models mirroring [1]_, use the BaggingClassifier:
    >>> from sktime.classification.ensemble import BaggingClassifier
    >>> from sktime.classification.deep_learning.inceptiontime import (
    ...     InceptionTimeClassifierTorch
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")  # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test")  # doctest: +SKIP
    >>> clf = BaggingClassifier(
    ...     InceptionTimeClassifierTorch(),
    ...     n_estimators=5,
    ...     bootstrap=False
    ... )  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    BaggingClassifier(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["hfawaz", "james-large", "noxthot"],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "InceptionTimeClassifierTorch",
        num_epochs: int = 1500,
        n_filters: int = 32,
        batch_size: int = 64,
        kernel_size: int = 40,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
        depth: int = 6,
        activation: str | None = None,
        activation_hidden: str = "relu",
        activation_inception: str | None = None,
        optimizer: str | None | Callable = "Adam",
        optimizer_kwargs: dict | None = None,
        criterion: str | None | Callable = "CrossEntropyLoss",
        criterion_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = None,
        callback_kwargs: dict | None = None,
        lr: float = 0.001,
        weights_init: str | None = None,
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
        self.weights_init = weights_init
        self.verbose = verbose
        self.random_state = random_state

        # input_size and num_classes to be inferred from the data
        # and will be set in _build_network
        self.input_size = None
        self.num_classes = None

        super().__init__(
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            criterion=self.criterion,
            criterion_kwargs=self.criterion_kwargs,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            callbacks=self.callbacks,
            activation=self.activation,
            callback_kwargs=self.callback_kwargs,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _build_network(self, X, y):
        """Build the InceptionTime network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels for the classification task.

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
        self.num_classes = len(np.unique(y))

        X = X.transpose(0, 2, 1)
        _, self.input_size, _ = X.shape

        if self.n_classes_ == 1:
            warnings.warn(
                "The provided data passed to CNNClassifierTorch contains "
                "a single label. If this is not intentional, please check.",
                UserWarning,
            )

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
