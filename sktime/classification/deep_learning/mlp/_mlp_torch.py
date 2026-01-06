"""Multi Layer Perceptron (MLP) for classification in PyTorch."""

__all__ = ["MLPClassifierTorch"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.mlp import MLPNetworkTorch


class MLPClassifierTorch(BaseDeepClassifierPytorch):
    """Multi Layer Perceptron Network (MLP) in PyTorch for time series classification.

    Implements a simple MLP network, as in [1]_.

    Parameters
    ----------
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the fully connected output layer.
    activation_hidden : str, default = "relu"
        Activation function used in the hidden layers.
        List of available PyTorch activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
    dropout : float or tuple, default=(0.1, 0.2, 0.2, 0.3)
        The dropout rate for the hidden layers.
        If float, the same rate is used for all layers.
        If tuple, it must have length equal to number of hidden layers in the MLP,
        each element specifying the dropout rate for the corresponding hidden layer.
        Current implementation of the MLP has 4 hidden layers.
    num_epochs : int, default = 2000
        The number of epochs to train the model.
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "Adam"
        The optimizer to use for training the model. List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    batch_size : int, default = 16
        The size of each mini-batch during training.
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
    .. [1] Wang et al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.

    Examples
    --------
    >>> from sktime.classification.deep_learning.mlp import MLPClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = MLPClassifierTorch(num_epochs=50, batch_size=4) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    MLPClassifierTorch(...)
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
        self: "MLPClassifierTorch",
        # model specific
        activation: str | None | Callable = None,
        activation_hidden: str = "relu",
        dropout: float | tuple = (0.1, 0.2, 0.2, 0.3),
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
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.dropout = dropout
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
        """Build the MLP network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels for the classification task.

        Returns
        -------
        model : MLPNetworkTorch instance
            The constructed MLP network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )
        self.num_classes = len(np.unique(y))
        # n_instances, n_dims, n_timesteps = X.shape
        _, n_dims, n_timesteps = X.shape
        # Input shape for MLP: (series_length, n_dimensions)
        self.input_size = (n_timesteps, n_dims)
        return MLPNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            activation=self._validated_activation,
            activation_hidden=self.activation_hidden,
            use_bias=self.use_bias,
            dropout=self.dropout,
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
            "activation": None,
            "activation_hidden": "relu",
            "dropout": (0.1, 0.2, 0.2, 0.1),
            "num_epochs": 10,
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
            "activation": None,
            "activation_hidden": "relu",
            "dropout": 0.1,
            "num_epochs": 12,
            "batch_size": 6,
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
        return [params1, params2, params3]
