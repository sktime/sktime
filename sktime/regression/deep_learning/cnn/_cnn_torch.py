"""Time Convolutional Neural Network (CNN) for regression in PyTorch."""

__authors__ = ["sabasiddique1"]
__all__ = ["CNNRegressorTorch"]

from collections.abc import Callable

from sktime.networks.cnn import CNNNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class CNNRegressorTorch(BaseDeepRegressorTorch):
    """Time Convolutional Neural Network in PyTorch for time series regression.

    Zhao et al. 2017 uses sigmoid activation in the hidden layers.
    To obtain same behaviour, set activation_hidden to "sigmoid" (default).

    Adapted from the TensorFlow CNN regressor. Same defaults as the TensorFlow
    implementation where applicable.

    Parameters
    ----------
    kernel_size : int, default = 7
        Length of the 1D convolution window.
    avg_pool_size : int, default = 3
        Size of the average pooling windows.
    n_conv_layers : int, default = 2
        Number of convolutional plus average pooling layers.
    filter_sizes : array-like of int, default = None
        Filter sizes per conv layer. If None, uses [6, 12].
    activation_hidden : str, default = "sigmoid"
        Activation for hidden conv layers. One of "relu", "sigmoid".
    padding : str, default = "auto"
        Padding logic: "auto" (same if series_length < 60 else valid), "valid", "same".
    use_bias : bool, default = True
        Whether conv and linear layers use a bias vector.
    num_epochs : int, default = 100
        Number of epochs to train the model.
    batch_size : int, default = 16
        Size of each mini-batch (same default as TF CNN).
    optimizer : str or None or optimizer instance, default = "Adam"
        Optimizer to use (TF CNN uses Adam lr=0.01).
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments for the optimizer.
    lr : float, default = 0.01
        Learning rate (same default as TF CNN).
    criterion : str or None or loss instance, default = "MSELoss"
        Loss function for training (same as TF CNN default).
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments for the criterion.
    callbacks : None or str or tuple of str, default = "ReduceLROnPlateau"
        Learning rate schedulers supported as callbacks.
    callback_kwargs : dict or None, default = None
        Keyword arguments for the callbacks.
    verbose : bool, default = False
        Whether to print progress during training.
    random_state : int, default = 0
        Seed for reproducibility.

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics, 28(1):2017.

    Examples
    --------
    >>> from sktime.regression.deep_learning.cnn import CNNRegressorTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> reg = CNNRegressorTorch(n_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> reg.fit(X_train, y_train)  # doctest: +SKIP
    CNNRegressorTorch(...)
    """

    _tags = {
        "authors": ["sabasiddique1"],
        "maintainers": ["sabasiddique1"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "CNNRegressorTorch",
        kernel_size: int = 7,
        avg_pool_size: int = 3,
        n_conv_layers: int = 2,
        filter_sizes: list | None = None,
        activation_hidden: str = "sigmoid",
        padding: str = "auto",
        use_bias: bool = True,
        num_epochs: int = 100,
        batch_size: int = 16,
        optimizer: str | None | Callable = "Adam",
        optimizer_kwargs: dict | None = None,
        lr: float = 0.01,
        criterion: str | None | Callable = "MSELoss",
        criterion_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        callback_kwargs: dict | None = None,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        self.activation_hidden = activation_hidden
        self.padding = padding
        self.use_bias = use_bias
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.lr = lr
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs or {}
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.verbose = verbose
        self.random_state = random_state
        self.input_size = None
        self.series_length = None

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
        X : numpy.ndarray
            Input data, shape (n_instances, n_dims, series_length).

        Returns
        -------
        model : CNNNetworkTorch
            The constructed CNN network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}."
            )
        _, self.input_size, self.series_length = X.shape
        return CNNNetworkTorch(
            input_size=self.input_size,
            num_classes=1,
            kernel_size=self.kernel_size,
            avg_pool_size=self.avg_pool_size,
            n_conv_layers=self.n_conv_layers,
            filter_sizes=self.filter_sizes,
            activation="linear",
            activation_hidden=self.activation_hidden,
            padding=self.padding,
            series_length=self.series_length,
            bias=self.use_bias,
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
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {}
        params2 = {
            "num_epochs": 10,
            "batch_size": 4,
            "avg_pool_size": 4,
        }
        params3 = {
            "num_epochs": 12,
            "batch_size": 6,
            "kernel_size": 2,
            "n_conv_layers": 1,
            "padding": "valid",
        }
        params4 = {
            "kernel_size": 5,
            "padding": "same",
            "num_epochs": 8,
        }
        return [params1, params2, params3, params4]
