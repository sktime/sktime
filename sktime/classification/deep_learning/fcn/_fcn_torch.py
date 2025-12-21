"""Fully Convolutional Network (FCN) for classification in PyTorch."""

__authors__ = ["Ali-John"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.fcn import FCNNetworkTorch


class FCNClassifierTorch(BaseDeepClassifierPytorch):
    """Fully Convolutional Network (FCN) for time series classification in PyTorch.

    Adapted from the implementation of Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    For a drop-in replacement based on TensorFlow, see `FCNClassifier`.

    Parameters
    ----------
    n_conv_layers : int, default = 3
        Number of convolutional blocks in the network
    filter_sizes : list of int or None, default = None
        Number of filters for each convolutional layer. If None, defaults to
        [128, 256, 128] for 3 layers as specified in the original paper
    kernel_sizes : list of int or None, default = None
        Kernel sizes for each convolutional layer. If None, defaults to [8, 5, 3]
        for 3 layers as specified in the original paper
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = "softmax"
        Activation function used in the fully connected output layer.
        List of supported activation functions: 'sigmoid', 'softmax',
        'logsoftmax', 'logsigmoid'. If None, no activation is applied.
    activation_hidden : str, default = "relu"
        Activation function used in the convolutional layers.
        List of supported activation functions: 'relu', 'tanh', 'sigmoid', etc.
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
    callbacks : None or str or a tuple of str, default = None
        Currently only learning rate schedulers are supported as callbacks.
        If more than one scheduler is passed, they are applied sequentially in the
        order they are passed. If None, then no learning rate scheduler is used.
        Note: Since PyTorch learning rate schedulers need to be initialized with
        the optimizer object, we only accept the class name (str) of the scheduler here
        and do not accept an instance of the scheduler. As that can lead to errors
        and unexpected behavior.
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

    References
    ----------
    .. [1] Wang et al, Time series classification from scratch with
    deep neural networks: A strong baseline.
    2017 International Joint Conference on Neural Networks (IJCNN)

    Examples
    --------
    >>> from sktime.classification.deep_learning.fcn import FCNClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = FCNClassifierTorch(num_epochs=20, batch_size=4) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    FCNClassifierTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Ali-John"],
        "maintainers": ["Ali-John"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "FCNClassifierTorch",
        # FCN parameters
        n_conv_layers: int = 3,
        filter_sizes: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        activation: str | None | Callable = "softmax",
        activation_hidden: str = "relu",
        # base classifier specific
        num_epochs: int = 2000,
        batch_size: int = 16,
        optimizer: str | None | Callable = "Adam",
        criterion: str | None | Callable = "CrossEntropyLoss",
        callbacks: None | str | tuple[str, ...] = None,
        optimizer_kwargs: dict | None = None,
        criterion_kwargs: dict | None = None,
        callback_kwargs: dict | None = None,
        lr: float = 0.01,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.activation_hidden = activation_hidden

        # store base classifier parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_kwargs = criterion_kwargs
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # input_size and num_classes to be inferred from the data
        # and will be set in _build_network
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
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )

        # Extract dimensions from data
        # n_instances, n_dims, n_timesteps = X.shape
        self.num_classes = len(np.unique(y))
        _, self.input_size, _ = X.shape

        return FCNNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            n_conv_layers=self.n_conv_layers,
            filter_sizes=self.filter_sizes,
            kernel_sizes=self.kernel_sizes,
            activation=self._validated_activation,
            activation_hidden=self.activation_hidden,
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
        params1 = {
            "num_epochs": 10,
            "batch_size": 4,
        }
        params2 = {
            "num_epochs": 12,
            "batch_size": 6,
            "activation": None,
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
            "num_epochs": 10,
            "batch_size": 4,
            "activation": "sigmoid",
            "optimizer": "Adam",
            "criterion": "BCELoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.01,
            "verbose": False,
            "random_state": 0,
        }  # functionally equivalent to params1 for binary classification
        params4 = {
            "num_epochs": 10,
            "batch_size": 4,
            "activation": None,
            "optimizer": "Adam",
            "criterion": "BCEWithLogitsLoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.01,
            "verbose": False,
            "random_state": 0,
        }  # functionally equivalent to params1 for binary classification
        params5 = {
            "num_epochs": 10,
            "batch_size": 4,
            "activation": "logsoftmax",
            "optimizer": "Adam",
            "criterion": "NLLLoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.01,
            "verbose": False,
            "random_state": 0,
        }  # functionally equivalent to params1 for multi-class classification
        return [params1, params2, params3, params4, params5]
