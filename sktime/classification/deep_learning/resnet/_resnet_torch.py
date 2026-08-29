"""Residual Network (ResNet) for classification in PyTorch."""

__authors__ = ["srupat"]
__all__ = ["ResNetClassifierTorch"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.resnet import ResNetNetworkTorch


class ResNetClassifierTorch(BaseDeepClassifierPytorch):
    """Residual neural network classifier in PyTorch, as described in [1].

    Adapted from the implementation from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    Parameters
    ----------
    n_filters : tuple of int, default = (64, 128, 128)
        Number of convolutional filters in each residual block. The length of
        this tuple determines the number of residual blocks. Any length >= 1
        is allowed.
    kernel_size : tuple of int, default = (8, 5, 3)
        Length of the 1D convolution window for the conv layers within a
        residual block, shared across all residual blocks. Any length >= 1
        is allowed.
    activation : str, Callable, or None, default=None
        Activation applied to the output layer.

        Permitted values:

        - ``None``: no activation is applied to the output layer and the network
          returns raw outputs (logits). This is typically required when using
          ``CrossEntropyLoss``, which expects logits as input.
        - ``str``: name of a class in ``torch.nn``. Case-sensitive names are
          recommended and must match PyTorch (e.g., ``"ReLU"``, ``"LeakyReLU"``).
          Lowercase aliases for common activations are also accepted
          (e.g., ``"relu"`` is resolved to ``"ReLU"``). The class is instantiated
          with default constructor arguments. Must be a valid ``torch.nn``
          activation; see
          https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        - ``torch.nn.Module``: an instance of a ``torch.nn.Module`` subclass,
          for example ``torch.nn.ReLU()``. Arbitrary callables are not supported.

    activation_hidden : str, Callable, or None, default="ReLU"
        Activation applied to the hidden layers, and after each residual
        connection.

        Permitted values:

        - ``None``: no activation is applied to the hidden layers.
        - ``str``: name of a class in ``torch.nn``. Case-sensitive names are
          recommended and must match PyTorch (e.g., ``"ReLU"``, ``"LeakyReLU"``).
          Lowercase aliases for common activations are also accepted
          (e.g., ``"relu"`` is resolved to ``"ReLU"``). The class is instantiated
          with default constructor arguments. Must be a valid ``torch.nn``
          activation; see
          https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        - ``torch.nn.Module``: an instance of a ``torch.nn.Module`` subclass,
          for example ``torch.nn.ReLU()``. Arbitrary callables are not supported.

    init_weights : bool, default = True
        Whether to apply custom initialization to the weights.
    num_epochs : int, default = 100
        The number of epochs to train the model.
    batch_size : int, default = 1
        The size of each mini-batch during training.
    optimizer : str or None or an instance of optimizers
        defined in torch.optim, default = "RMSprop"
        The optimizer to use for training the model.
        List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    criterion : str or None or an instance of a loss function
        defined in PyTorch, default = "CrossEntropyLoss"
        The loss function to be used in training the neural network.
        List of available loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions
    callbacks : None or str or a tuple of str, default = "ReduceLROnPlateau"
        Learning rate schedulers applied during training.
        Currently only learning rate schedulers are supported as callbacks.
        If more than one scheduler is passed, they are applied sequentially in the
        order they are passed. If None, then no learning rate scheduler is used.
        Note: Since PyTorch learning rate schedulers need to be initialized with
        the optimizer object, we only accept the class name (str) of the scheduler here
        and do not accept an instance of the scheduler. As that can lead to errors
        and unexpected behavior.
        List of available learning rate schedulers:
        https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the loss function.
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
    metrics : None or str or Callable or tuple of str and/or Callable, default = None
        Metrics to compute during training. If None, no metrics are computed beyond
        the loss. Metrics are computed from torchmetrics library.
        If a string/Callable is passed, it must be one of the metrics defined in
        https://lightning.ai/docs/torchmetrics/stable/
        Examples: "Accuracy", "F1Score", "Precision", "Recall"
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Wang et al, Time series classification from scratch with deep neural
    networks: A strong baseline, International joint conference on neural
    networks (IJCNN), 2017.

    Examples
    --------
    >>> from sktime.classification.deep_learning.resnet import ResNetClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = ResNetClassifierTorch(num_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    ResNetClassifierTorch(...)
    """

    _tags = {
        "authors": ["srupat"],
        "maintainers": ["srupat"],
        "python_version": ">=3.10, <3.15",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
        "capability:multivariate": True,
    }

    def __init__(
        self: "ResNetClassifierTorch",
        # model specific
        n_filters: tuple[int, ...] = (64, 128, 128),
        kernel_size: tuple[int, ...] = (8, 5, 3),
        activation: str | Callable | None = None,
        activation_hidden: str | Callable = "ReLU",
        init_weights: bool = True,
        # base classifier specific
        num_epochs: int = 100,
        batch_size: int = 1,
        optimizer: str | None | Callable = "RMSprop",
        criterion: str | None | Callable = "CrossEntropyLoss",
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        optimizer_kwargs: dict | None = None,
        criterion_kwargs: dict | None = None,
        callback_kwargs: dict | None = None,
        metrics: None | str | Callable | tuple[str | Callable, ...] = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.init_weights = init_weights
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.metrics = metrics
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

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
            metrics=self.metrics,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        # input_size and num_classes inferred from the data and will be
        # set in _build_network
        self.input_size = None
        self.num_classes = None

        super().__post_init__()

    def _build_network(self, X, y):
        """Build the ResNet network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels corresponding to the input data.

        Returns
        -------
        model : ResNetNetworkTorch
             An instance of the ResNetNetworkTorch class initialized with the
             appropriate parameters.
        """
        self.num_classes = len(np.unique(y))
        self.input_size = X.shape
        return ResNetNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            activation=self._callable_activations["activation"],
            activation_hidden=self._callable_activations["activation_hidden"],
            random_state=self.random_state,
            init_weights=self.init_weights,
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
            "n_filters": (8, 16),
            "kernel_size": (3, 2),
            "num_epochs": 20,
            "batch_size": 4,
        }
        params3 = {
            "n_filters": (16,),
            "kernel_size": (5, 3),
            "activation_hidden": "LeakyReLU",
            "num_epochs": 2,
            "random_state": 42,
        }
        return [params1, params2, params3]
