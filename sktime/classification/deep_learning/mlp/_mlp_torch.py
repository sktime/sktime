"""MLP (Multi-Layer Perceptron) Classifier for Time Series Classification in PyTorch."""

__authors__ = ["RecreationalMath"]
__all__ = ["MLPClassifierTorch"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.mlp import MLPNetworkTorch


class MLPClassifierTorch(BaseDeepClassifierPytorch):
    """Multi Layer Perceptron classifier in PyTorch for time series classification.

    A simple MLP classifier based on MLP network defined in [1]_.
    Adapts the implementation from [2]_ and [3]_.

    Parameters
    ----------
    hidden_dim : int, default = 500
        Dimensionality of the hidden layers.
    n_layers : int, default = 4
        Number of hidden layers.
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the fully connected output layer. List of supported
        activation functions: ['sigmoid', 'softmax', 'logsoftmax', 'logsigmoid'].
        If None, then no activation function is applied.
    activation_hidden : str or None or an instance of activation functions defined in
        torch.nn, default = "relu"
        The activation function applied inside the hidden layers of the MLP.
        Can be any of "relu", "leakyrelu", "elu", "prelu", "gelu", "selu",
        "rrelu", "celu", "tanh", "hardtanh".
    bias : bool, default = True
        If False, then the layer does not use bias weights.
    dropout : float or tuple of floats, default = (0.1, 0.2, 0.2, 0.3)
        If dropout is a non-zero float, it introduces a Dropout layer on the outputs
        of each hidden layer of the MLP, with dropout probability equal to dropout.
        If dropout is a tuple of floats, it must have length equal to n_layers, and
        each element specifies the dropout probability in the corresponding hidden
        layer of the MLP.
    fc_dropout : float, default = 0.0
        If non-zero, introduces a Dropout layer on the outputs of the fully
        connected output layer, with dropout probability equal to fc_dropout.
    num_epochs : int, default = 100
        The number of epochs to train the model.
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "Adam"
        The optimizer to use for training the model. List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    batch_size : int, default = 1
        The size of each mini-batch during training.
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
        Note: Since PyTorch learning rate schedulers need to be initialized with
        the optimizer object, we only accept the class name (str) of the scheduler here
        and do not accept an instance of the scheduler. As that can lead to errors
        and unexpected behavior.
        List of available learning rate schedulers:
        https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1]  Network originally defined in:
    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }
    .. [2] Deep learning for time series classification: a review Fawaz et al. 2019
    .. [3] source code for [2]
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

    Examples
    --------
    >>> from sktime.classification.deep_learning.mlp import MLPClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = MLPClassifierTorch(n_epochs=50,batch_size=2) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    MLPClassifierTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "MLPClassifierTorch",
        # model architecture parameters
        hidden_dim: int = 500,
        n_layers: int = 4,
        activation: str | None | Callable = None,
        activation_hidden: str | None | Callable = "relu",
        bias: bool = True,
        dropout: float | tuple[float, ...] = (0.1, 0.2, 0.2, 0.3),
        fc_dropout: float = 0.0,
        # base classifier parameters
        num_epochs: int = 100,
        optimizer: str | None | Callable = "Adam",
        optimizer_kwargs: dict | None = None,
        batch_size: int = 1,
        criterion: str | None | Callable = "CrossEntropyLoss",
        criterion_kwargs: dict | None = None,
        callbacks: str | tuple[str] | None = "ReduceLROnPlateau",
        callback_kwargs: dict | None = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.bias = bias
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # input_shape and num_classes to be inferred from the data
        # and will be set in _build_network
        self.input_shape = None
        self.num_classes = None

        super().__init__(
            num_epochs=self.num_epochs,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            batch_size=self.batch_size,
            activation=self.activation,
            criterion=self.criterion,
            criterion_kwargs=self.criterion_kwargs,
            callbacks=self.callbacks,
            callback_kwargs=self.callback_kwargs,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _build_network(self, X, y):
        """Build the RNN network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels for the classification task.

        Returns
        -------
        model : RNNNetworkTorch instance
            The constructed RNN network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )

        # n_instances, n_dims, n_timesteps = X.shape
        self.num_classes = len(np.unique(y))
        self.input_shape = X.shape
        return MLPNetworkTorch(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            activation=self._validated_activation,  # use self._validated_activation
            activation_hidden=self.activation_hidden,
            bias=self.bias,
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
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
            "hidden_dim": 5,
            "n_layers": 1,
            "activation": None,
            "activation_hidden": "relu",
            "bias": False,
            "dropout": 0.0,
            "fc_dropout": 0.0,
            "num_epochs": 50,
            "batch_size": 2,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }
        params3 = {
            "hidden_dim": 5,
            "n_layers": 1,
            "activation": "sigmoid",
            "activation_hidden": "relu",
            "bias": False,
            "dropout": 0.0,
            "fc_dropout": 0.0,
            "num_epochs": 50,
            "batch_size": 2,
            "optimizer": "Adam",
            "criterion": "BCELoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }  # functionally equivalent to params2 for binary classification
        params4 = {
            "hidden_dim": 5,
            "n_layers": 1,
            "activation": None,
            "activation_hidden": "relu",
            "bias": False,
            "dropout": 0.0,
            "fc_dropout": 0.0,
            "num_epochs": 50,
            "batch_size": 2,
            "optimizer": "Adam",
            "criterion": "BCEWithLogitsLoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }  # functionally equivalent to params2 for binary classification
        params5 = {
            "hidden_dim": 5,
            "n_layers": 1,
            "activation": "logsoftmax",
            "activation_hidden": "relu",
            "bias": False,
            "dropout": 0.0,
            "fc_dropout": 0.0,
            "num_epochs": 50,
            "batch_size": 2,
            "optimizer": "Adam",
            "criterion": "NLLLoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }  # functionally equivalent to params2 for multi-class classification
        return [params1, params2, params3, params4, params5]
