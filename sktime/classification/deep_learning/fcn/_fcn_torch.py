"""Fully Convolutional Network (FCN) for classification in PyTorch."""

__authors__ = ["kajal-jotwani"]
__all__ = ["FCNClassifierTorch"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.fcn import FCNNetworkTorch


class FCNClassifierTorch(BaseDeepClassifierPytorch):
    """Fully Convolutional Network (FCN) in PyTorch for time series classification.

    Adapted from the TensorFlow FCN implementation in sktime.

    Parameters
    ----------
    filter_sizes : tuple of int, default = (128, 256, 128)
        Number of filters for each convolutional layer. The number of
        convolutional layers is inferred from the length of this tuple.
    kernel_sizes : tuple of int, default = (8, 5, 3)
        Kernel size for each convolutional layer. Must have the same length
        as ``filter_sizes``.
    activation_hidden : str or None or an instance of activation functions defined in
        torch.nn, default = "relu"
        Activation function applied after each BatchNorm layer in the
        convolutional blocks.
        If str, supported values are ``"relu"``, ``"tanh"``, ``"sigmoid"``.
        If not str, must be an instantiated ``torch.nn.Module`` activation.
        If ``None``, no activation is applied (identity).
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the fully connected output layer.
    init_weights : str or None, default = "kaiming_uniform"
        The method to initialize the weights of the convolutional layers.
        Supported values: ``"kaiming_uniform"``, ``"kaiming_normal"``,
        ``"xavier_uniform"``, ``"xavier_normal"``, or ``None`` for default
        PyTorch initialization.
    num_epochs : int, default = 2000
        The number of epochs to train the model.
    batch_size : int, default = 16
        The size of each mini-batch during training.
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "Adam"
        The optimizer to use for training the model. List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    criterion : case insensitive str or None or an instance of a loss function
        defined in PyTorch, default = "CrossEntropyLoss"
        The loss function to be used in training the neural network.
        List of available loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the loss function.
    callbacks : None or str or a tuple of str, default = "ReduceLROnPlateau"
        Currently only learning rate schedulers are supported as callbacks.
        If more than one scheduler is passed, they are applied sequentially in the
        order they are passed. If None, then no learning rate scheduler is used.
        List of available learning rate schedulers:
        https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
    lr : float, default = 0.01
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    Examples
    --------
    >>> from sktime.classification.deep_learning.fcn import FCNClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = FCNClassifierTorch(num_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    FCNClassifierTorch(...)
    """

    _tags = {
        "authors": ["kajal-jotwani"],
        "maintainers": ["kajal-jotwani"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "FCNClassifierTorch",
        filter_sizes: tuple = (128, 256, 128),
        kernel_sizes: tuple = (8, 5, 3),
        activation_hidden="relu",  # str, torch.nn.Module instance, or None
        activation: str | None | Callable = None,
        init_weights: str | None = "kaiming_uniform",
        num_epochs: int = 2000,
        batch_size: int = 16,
        optimizer: str | None | Callable = "Adam",
        optimizer_kwargs: dict | None = None,
        criterion: str | None | Callable = "CrossEntropyLoss",
        criterion_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        callback_kwargs: dict | None = None,
        lr: float = 0.01,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.activation_hidden = activation_hidden
        self.activation = activation
        self.init_weights = init_weights
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state
        self.input_size = None
        self.num_classes = None

        super().__init__(
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            activation=self.activation,
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
        """Build the FCN network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels for the classification task.

        Returns
        -------
        model : FCNNetworkTorch instance
            The constructed FCN network.
        """
        self.num_classes = len(np.unique(y))
        _, self.input_size, _ = X.shape

        return FCNNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            filter_sizes=self.filter_sizes,
            kernel_sizes=self.kernel_sizes,
            activation_hidden=self.activation_hidden,
            activation=self._validated_activation,
            init_weights=self.init_weights,
            random_state=self.random_state,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

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
            "filter_sizes": (32, 64, 32),
            "kernel_sizes": (4, 3, 2),
            "activation_hidden": "relu",
            "activation": None,
            "init_weights": "kaiming_uniform",
            "num_epochs": 50,
            "batch_size": 4,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.01,
            "verbose": False,
            "random_state": 0,
        }
        params3 = {
            "filter_sizes": (32, 64, 32),
            "kernel_sizes": (4, 3, 2),
            "activation": "sigmoid",
            "num_epochs": 50,
            "batch_size": 4,
            "optimizer": "Adam",
            "criterion": "BCELoss",
            "callbacks": None,
            "lr": 0.01,
            "random_state": 42,
        }
        params4 = {
            "filter_sizes": (32, 64, 32),
            "kernel_sizes": (4, 3, 2),
            "activation": None,
            "num_epochs": 50,
            "batch_size": 4,
            "optimizer": "Adam",
            "criterion": "BCEWithLogitsLoss",
            "callbacks": None,
            "lr": 0.01,
            "random_state": 0,
        }
        params5 = {
            "filter_sizes": (32, 64, 32),
            "kernel_sizes": (4, 3, 2),
            "activation": "logsoftmax",
            "num_epochs": 50,
            "batch_size": 4,
            "optimizer": "Adam",
            "criterion": "NLLLoss",
            "callbacks": None,
            "lr": 0.01,
            "random_state": 0,
        }
        return [params1, params2, params3, params4, params5]
