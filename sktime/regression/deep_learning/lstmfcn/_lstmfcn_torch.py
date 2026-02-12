"""LSTM-FCN regressor for time series regression in PyTorch."""

__authors__ = ["Faakhir30"]
__all__ = ["LSTMFCNRegressorTorch"]

from collections.abc import Callable

from sktime.networks.lstmfcn import LSTMFCNNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class LSTMFCNRegressorTorch(BaseDeepRegressorTorch):
    """LSTM-FCN regressor for time series regression in PyTorch.

    Combines an LSTM arm with a CNN arm. Optionally uses an attention mechanism in the
    LSTM which the author indicates provides improved performance.

    Parameters
    ----------
    kernel_sizes : tuple of int, default=(8, 5, 3)
        Specifying the length of the 1D convolution windows for each conv layer
    filter_sizes : tuple of int, default=(128, 256, 128)
        Size of filter for each conv layer
    lstm_size : int, default=8
        Output dimension for LSTM layer (hidden state size)
    dropout : float, default=0.8
        Controls dropout rate of LSTM layer
    attention : bool, default=False
        If True, uses attention mechanism before LSTM layer
    activation : str or None, default=None
        Activation function used in the output layer.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    activation_hidden : str, default="relu"
        Activation function used for hidden layers.
        Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'
    num_epochs : int, default=2000
        The number of epochs to train the model.
    batch_size : int, default=128
        The size of each mini-batch during training.
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "SGD"
        The optimizer to use for training the model. List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    criterion : case insensitive str or None or an instance of a loss function
        defined in PyTorch, default = "MSELoss"
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
    init_weights: str or None, default = 'kaiming_uniform'
        The method to initialize the weights of the conv layers. Supported values are
        'kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', or None
        for default PyTorch initialization.
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.
    random_state : int or None, default = None
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Karim et al. Multivariate LSTM-FCNs for Time Series Classification, 2019
    https://arxiv.org/pdf/1801.04503.pdf

    Examples
    --------
    >>> from sktime.regression.deep_learning.lstmfcn import LSTMFCNRegressorTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> reg = LSTMFCNRegressorTorch(num_epochs=50, batch_size=2)  # doctest: +SKIP
    >>> reg.fit(X_train, y_train)  # doctest: +SKIP
    LSTMFCNRegressorTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["jnrusson1", "solen0id", "nilesh05apr", "noxthot"],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "LSTMFCNRegressorTorch",
        # model specific
        kernel_sizes: tuple = (8, 5, 3),
        filter_sizes: tuple = (128, 256, 128),
        lstm_size: int = 8,
        dropout: float = 0.8,
        attention: bool = False,
        activation: str | None = None,
        activation_hidden: str = "relu",
        # base regressor specific
        num_epochs: int = 2000,
        batch_size: int = 128,
        optimizer: str | None | Callable = "SGD",
        optimizer_kwargs: dict | None = None,
        criterion: str | None | Callable = "MSELoss",
        criterion_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        callback_kwargs: dict | None = None,
        init_weights: str | None = "kaiming_uniform",
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.attention = attention
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
        self.random_state = random_state
        self.init_weights = init_weights

        # input_size to be inferred from the data
        # and will be set in _build_network
        self.input_size = None
        self.num_classes = 1  # because regression

        if len(self.filter_sizes) != len(self.kernel_sizes):
            raise ValueError(
                f"Length of `filter_sizes` {len(self.filter_sizes)} must match "
                f"the number of convolutional layers determined by the length of tuple "
                f"`kernel_sizes` {len(self.kernel_sizes)}."
            )

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
        """Build the LSTM-FCN network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.

        Returns
        -------
        model : LSTMFCNNetworkTorch instance
            The constructed LSTM-FCN network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )
        # n_instances, n_dims, n_timesteps = X.shape
        _, self.input_size, _ = X.shape
        return LSTMFCNNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            kernel_sizes=self.kernel_sizes,
            filter_sizes=self.filter_sizes,
            lstm_size=self.lstm_size,
            dropout=self.dropout,
            attention=self.attention,
            activation=self.activation,
            activation_hidden=self.activation_hidden,
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

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {"num_epochs": 30, "batch_size": 8}
        params2 = {
            "kernel_sizes": (3, 2, 1),
            "filter_sizes": (8, 16, 8),
            "lstm_size": 4,
            "dropout": 0.5,
            "attention": False,
            "activation": "relu",
            "num_epochs": 50,
            "batch_size": 2,
            "optimizer": "RMSprop",
            "criterion": "MSELoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }
        params3 = {
            "kernel_sizes": (3, 2, 1),
            "filter_sizes": (8, 16, 8),
            "lstm_size": 4,
            "dropout": 0.25,
            "attention": True,
            "activation": "relu",
            "num_epochs": 50,
            "batch_size": 2,
            "optimizer": "RMSprop",
            "criterion": "MSELoss",
            "callbacks": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }

        return [params1, params2, params3]
