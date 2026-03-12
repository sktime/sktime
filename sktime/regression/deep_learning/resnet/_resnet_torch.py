"""Residual Network (ResNet) for regression in PyTorch."""

__authors__ = ["DCchoudhury15"]
__all__ = ["ResNetRegressorTorch"]

from collections.abc import Callable

from sktime.networks.resnet import ResNetNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class ResNetRegressorTorch(BaseDeepRegressorTorch):
    """Residual Neural Network for time series regression in PyTorch.

    Parameters
    ----------
    n_filters_per_block : tuple of int, default = (64, 128, 128)
        Number of convolutional filters in each residual block.
        The length of this tuple determines the number of residual blocks.
    kernel_sizes : tuple of int, default = (8, 5, 3)
        Kernel sizes of the 3 Conv1d layers within each residual block.
        Must be a tuple of length 3.
    padding : str, default = "same"
        Padding mode for all Conv1d layers.
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = None
        Activation function used in the output layer.
        If None, no activation is applied (raw values are returned).
    activation_hidden : str, default = "relu"
        Activation function used in the residual blocks.
        Supported values: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu',
        'selu', 'gelu'.
    use_bias : bool, default = True
        Whether the final fully connected layer uses a bias vector.
    init_weights : bool, default = True
        Whether to apply weight initialization to Conv1d layers.
        If True, applies ``kaiming_uniform_`` initialization
        (appropriate for ReLU-based convolutional networks).
    num_epochs : int, default = 1500
        The number of epochs to train the model.
    batch_size : int, default = 16
        The size of each mini-batch during training.
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "Adam"
        The optimizer to use for training the model. List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    criterion : case insensitive str or None or an instance of a loss function
        defined in PyTorch, default = "MSELoss"
        The loss function to be used in training the neural network.
        List of available loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions
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
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the loss function.
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
    .. [1] Wang et al, Time series classification from scratch with deep
        neural networks: A strong baseline, IJCNN 2017.

    Examples
    --------
    >>> from sktime.regression.deep_learning.resnet import ResNetRegressorTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> reg = ResNetRegressorTorch(num_epochs=20, batch_size=4) # doctest: +SKIP
    >>> reg.fit(X_train, y_train) # doctest: +SKIP
    ResNetRegressorTorch(...)
    """

    _tags = {
        "authors": ["DCchoudhury15"],
        "maintainers": ["DCchoudhury15"],
        "python_version": ">=3.10,<3.15",
        "python_dependencies": "torch",
        "capability:multivariate": True,
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "ResNetRegressorTorch",
        # model specific
        n_filters_per_block: tuple = (64, 128, 128),
        kernel_sizes: tuple = (8, 5, 3),
        padding: str = "same",
        activation: str | None | Callable = None,
        activation_hidden: str = "relu",
        use_bias: bool = True,
        init_weights: bool = True,
        # base regressor specific
        num_epochs: int = 1500,
        batch_size: int = 16,
        optimizer: str | None | Callable = "Adam",
        criterion: str | None | Callable = "MSELoss",
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        optimizer_kwargs: dict | None = None,
        criterion_kwargs: dict | None = None,
        callback_kwargs: dict | None = None,
        lr: float = 0.01,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.n_filters_per_block = n_filters_per_block
        self.kernel_sizes = kernel_sizes
        self.padding = padding
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.use_bias = use_bias
        self.init_weights = init_weights
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

        if len(self.kernel_sizes) != 3:
            raise ValueError(
                "ResNet residual blocks always contain exactly 3 Conv1d layers. "
                f"`kernel_sizes` must be a tuple of length 3, "
                f"but got length {len(self.kernel_sizes)}."
            )

        self.input_size = None
        self.num_classes = 1

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
        """Build the ResNet network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.

        Returns
        -------
        model : ResNetNetworkTorch instance
            The constructed ResNet network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )
        _, self.input_size, _ = X.shape
        return ResNetNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            n_filters_per_block=self.n_filters_per_block,
            kernel_sizes=self.kernel_sizes,
            padding=self.padding,
            activation=self.activation,
            activation_hidden=self.activation_hidden,
            use_bias=self.use_bias,
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
        params1 = {
            "num_epochs": 6,
            "batch_size": 4,
        }
        params2 = {
            "num_epochs": 4,
            "batch_size": 6,
            "n_filters_per_block": (16, 32),
            "kernel_sizes": (3, 3, 3),
            "use_bias": False,
            "random_state": 42,
        }
        return [params1, params2]
