"""Time Convolutional Neural Network (CNN) for regression in PyTorch."""

__all__ = ["CNNRegressorTorch"]


from collections.abc import Callable

from sktime.networks.cnn import CNNNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class CNNRegressorTorch(BaseDeepRegressorTorch):
    """Time Convolutional Neural Network (CNN) in PyTorch, as described in [1].

    Zhao et al. 2017 uses sigmoid activation in the hidden layers.

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    Parameters
    ----------
    num_epochs : int, default = 2000
        Number of epochs to train the model.
    batch_size : int, default = 16
        Size of each mini-batch.
    kernel_sizes : tuple of int, default = (7, 7)
        A tuple of length equal to the number of conv layers with each entry in
        the tuple specifies the kernel size for the corresponding convolutional
        layer. The length of ``kernel_sizes`` must be equal to the length of
        ``filter_sizes``.
    avg_pool_size : int, default = 3
        Size of the average pooling window.
    filter_sizes : tuple of int, default = (6, 12)
        A tuple of length equal to the number of conv layers with each entry in
        the tuple specifies the filter size for the corresponding convolutional
        layer. The length of ``filter_sizes`` must be equal to the length of
        ``kernel_sizes``.
    use_bias : bool, default = True
        Whether to use bias in output layer.
    padding : string, default = "auto"
        Controls padding logic for the convolutional layers,
        i.e. whether ``'valid'`` and ``'same'`` are passed to the ``Conv1D`` layer.
        - "auto": as per original implementation, ``"same"`` is passed if
          ``input_shape[0] < 60`` in the input layer, and ``"valid"`` otherwise.
        - "valid", "same", and other values are passed directly to ``Conv1D``
    activation : str, Callable, or None, default=None
        Activation applied to the output layer.

        Permitted values:

        - ``None``: no activation is applied to the output layer and the network
          returns raw outputs.
        - ``str``: name of a class in ``torch.nn``. Case-sensitive names are
          recommended and must match PyTorch (e.g., ``"ReLU"``, ``"LeakyReLU"``).
          Lowercase aliases for common activations are also accepted
          (e.g., ``"relu"`` is resolved to ``"ReLU"``). The class is instantiated
          with default constructor arguments. Must be a valid ``torch.nn``
          activation; see
          https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        - ``torch.nn.Module``: an instance of a ``torch.nn.Module`` subclass,
          for example ``torch.nn.ReLU()``. Arbitrary callables are not supported.

    activation_hidden : str, Callable, or None, default="Sigmoid"
        Activation applied to the hidden layers.

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

        Recommended activations: ``Sigmoid``, ``ReLU``, ``Tanh``.
    optimizer : str or callable, default = "Adam"
        Optimizer to use. Same as TF default (Adam).
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments for the optimizer.
    criterion : str or callable, default = "MSELoss"
        Loss function (TF uses mean_squared_error).
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments for the criterion.
    callbacks : None or str or tuple of str, default = "ReduceLROnPlateau"
        Learning rate schedulers as callbacks.
    callback_kwargs : dict or None, default = None
        Keyword arguments for callbacks.
    lr : float, default = 0.01
        Learning rate (TF CNN uses Adam(lr=0.01)).
    verbose : bool, default = False
        Whether to print progress during training.
    init_weights: str or None, default = None
        The method to initialize the weights of the conv layers. Supported values are
        'kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', or None
        for default PyTorch initialization.
    random_state : int or None, default = None
        Seed for reproducibility.

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
       Journal of Systems Engineering and Electronics, 28(1):2017.

    Examples
    --------
    >>> from sktime.regression.deep_learning.cnn import CNNRegressorTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>> reg = CNNRegressorTorch(num_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> reg.fit(X_train, y_train)  # doctest: +SKIP
    CNNRegressorTorch(...)
    """

    _tags = {
        "authors": ["hfawaz", "AurumnPegasus", "achieveordie", "noxthot", "Faakhir30"],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "CNNRegressorTorch",
        num_epochs: int = 2000,
        batch_size: int = 16,
        kernel_sizes: tuple[int, ...] = (7, 7),
        avg_pool_size: int = 3,
        filter_sizes: tuple[int, ...] = (6, 12),
        use_bias: bool = True,
        padding: str = "auto",
        activation: str | Callable | None = None,
        activation_hidden: str | Callable = "Sigmoid",
        optimizer: str | None | Callable = "Adam",
        optimizer_kwargs: dict | None = None,
        criterion: str | None | Callable = "MSELoss",
        criterion_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        callback_kwargs: dict | None = None,
        lr: float = 0.01,
        verbose: bool = False,
        init_weights: str | None = None,
        random_state: int | None = None,
    ):
        self.kernel_sizes = kernel_sizes
        self.avg_pool_size = avg_pool_size
        self.filter_sizes = filter_sizes
        self.activation_hidden = activation_hidden
        self.activation = activation
        self.padding = padding
        self.use_bias = use_bias
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.verbose = verbose
        self.init_weights = init_weights
        self.random_state = random_state

        if len(filter_sizes) != len(kernel_sizes):
            raise ValueError(
                f"Length of filter_sizes ({len(filter_sizes)}) must match "
                f"length of kernel_sizes ({len(kernel_sizes)}) in CNNRegressorTorch."
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
        """Build the CNN network.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_instances, n_dims, series_length).

        Returns
        -------
        CNNNetworkTorch
            The constructed CNN network.
        """
        # X arrives in sktime format: (n_instances, n_dims, n_timesteps)
        # The base class's _build_dataloader transposes it to
        # (batch, n_timesteps, n_dims) before passing to forward().
        # But at this point, X has not been transposed.
        # So input_size = n_dims is correct here
        n_dims = X.shape[1]
        series_length = X.shape[2]
        input_shape = (n_dims, series_length)

        return CNNNetworkTorch(
            input_shape=input_shape,
            num_classes=1,
            kernel_sizes=self.kernel_sizes,
            avg_pool_size=self.avg_pool_size,
            filter_sizes=self.filter_sizes,
            use_bias=self.use_bias,
            activation=self._callable_activations["activation"],
            activation_hidden=self._callable_activations["activation_hidden"],
            padding=self.padding,
            init_weights=self.init_weights,
            random_state=self.random_state,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "num_epochs": 10,
            "batch_size": 4,
            "avg_pool_size": 4,
            "activation_hidden": "ReLU",
        }
        params2 = {
            "num_epochs": 12,
            "batch_size": 6,
            "kernel_sizes": (2,),
            "filter_sizes": (4,),
        }
        return [params1, params2]
