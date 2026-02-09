"""Multi-scale Attention Convolutional Neural Network (MACNN) Classifier in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["MACNNClassifierTorch"]

import warnings
from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.macnn import MACNNNetworkTorch


class MACNNClassifierTorch(BaseDeepClassifierPytorch):
    """Multi-Scale Attention Convolutional Neural Network (MACNN) classifier in PyTorch.

    This classifier implements a multi-scale attention mechanism that learns feature
    representations across different temporal scales.

    Parameters
    ----------
    padding : str, default="same"
        The type of padding to be provided in MACNN Blocks.
        Used for pooling layers only. Convolution layers always use "same" padding,
        so that multi-scale outputs can be concatenated.
    pool_size : int, default=3
        A single value representing pooling windows which are applied
        between two MACNN Blocks.
    strides : int, default=2
        A single value representing strides to be taken during the
        pooling operation.
    repeats : int, default=2
        The number of MACNN Blocks to be stacked.
    filter_sizes : tuple of int, default=(64, 128, 256)
        The filter sizes of Conv1D layers within each MACNN Block.
    kernel_size : tuple of int, default=(3, 6, 12)
        The kernel sizes of Conv1D layers within each MACNN Block.
    reduction : int, default=16
        The factor by which the first dense layer of a MACNN Block will be divided by.
    activation : str, default=None
        Activation function used for final output layer.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    activation_hidden : str, default="relu"
        Activation function used for the hidden layers.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    num_epochs : int, default=1500
        The number of epochs to train the model.
    batch_size : int, default=4
        The size of each mini-batch during training.
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "RMSprop"
        The optimizer to use for training the model.
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    criterion : case insensitive str or None or an instance of a loss function
        defined in PyTorch, default = "CrossEntropyLoss"
        The loss function to be used in training the neural network.
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the loss function.
    callbacks : None or str or a tuple of str, default = "ReduceLROnPlateau"
        Currently only learning rate schedulers are supported as callbacks.
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.
    weights_init: str or None, default = None
        The method to initialize the weights of the conv layers. Supported values are
        'kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', or None
        for default PyTorch initialization.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Wei Chen et. al, Multi-scale Attention Convolutional
    Neural Network for time series classification,
    Neural Networks, Volume 136, 2021, Pages 126-140, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.01.001.

    Examples
    --------
    >>> from sktime.classification.deep_learning.macnn import MACNNClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = MACNNClassifierTorch(num_epochs=50, batch_size=2)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    MACNNClassifierTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["jnrusson1", "noxthot"],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "MACNNClassifierTorch",
        # model specific
        padding: str = "same",
        pool_size: int = 3,
        strides: int = 2,
        repeats: int = 2,
        filter_sizes: tuple = (64, 128, 256),
        kernel_size: tuple = (3, 6, 12),
        reduction: int = 16,
        activation: str | None = None,
        activation_hidden: str = "relu",
        # base classifier specific
        num_epochs: int = 100,
        batch_size: int = 1,
        optimizer: str | None | Callable = "RMSprop",
        optimizer_kwargs: dict | None = None,
        criterion: str | None | Callable = "CrossEntropyLoss",
        criterion_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        callback_kwargs: dict | None = None,
        lr: float = 0.001,
        verbose: bool = False,
        weights_init: str | None = None,
        random_state: int = 0,
    ):
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides
        self.repeats = repeats
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.activation = activation
        self.activation_hidden = activation_hidden
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
        self.weights_init = weights_init
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
            callback_kwargs=self.callback_kwargs,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _build_network(self, X, y):
        """Build the MACNN network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels for the classification task.

        Returns
        -------
        model : MACNNNetworkTorch instance
            The constructed MACNN network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )
        # n_instances, n_dims, n_timesteps = X.shape
        self.num_classes = len(np.unique(y))
        _, self.input_size, _ = X.shape

        if self.num_classes == 1:
            warnings.warn(
                "The provided data passed to MACNNClassifierTorch contains "
                "a single label. If this is not intentional, please check.",
                UserWarning,
            )

        return MACNNNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            padding=self.padding,
            pool_size=self.pool_size,
            strides=self.strides,
            repeats=self.repeats,
            filter_sizes=self.filter_sizes,
            kernel_size=self.kernel_size,
            reduction=self.reduction,
            activation=self.activation,
            weights_init=self.weights_init,
            random_state=self.random_state,
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
        params1 = {"num_epochs": 2, "batch_size": 8}
        params2 = {
            "padding": "valid",
            "pool_size": 2,
            "strides": 1,
            "repeats": 1,
            "filter_sizes": (32, 64, 128),
            "kernel_size": (3, 6, 12),
            "reduction": 16,
            "num_epochs": 50,
            "batch_size": 4,
            "optimizer": "RMSprop",
            "criterion": "CrossEntropyLoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
            "weights_init": None,
        }
        params3 = {
            "padding": "same",
            "pool_size": 3,
            "strides": 2,
            "repeats": 2,
            "filter_sizes": (32, 64, 128),
            "kernel_size": (3, 6, 12),
            "reduction": 8,
            "num_epochs": 50,
            "batch_size": 4,
            "optimizer": "RMSprop",
            "criterion": "CrossEntropyLoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }

        return [params1, params2, params3]
