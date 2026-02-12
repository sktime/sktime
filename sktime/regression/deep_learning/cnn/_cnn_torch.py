"""Time Convolutional Neural Network (CNN) for regression in PyTorch."""

__all__ = ["CNNRegressorTorch"]


from sktime.networks.cnn import CNNNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class CNNRegressorTorch(BaseDeepRegressorTorch):
    """Time Convolutional Neural Network (CNN) in PyTorch, as described in [1].

    Zhao et al. 2017 uses sigmoid activation in the hidden layers.
    To obtain same behaviour as Zhao et al. 2017, set activation_hidden to "sigmoid".

    Adapted from the implementation from Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/cnn.py

    Parameters
    ----------
    num_epochs : int, default = 2000
        Number of epochs to train the model.
    n_conv_layers : int, default = 2
        Number of convolutional plus average pooling layers.
    batch_size : int, default = 16
        Size of each mini-batch.
    kernel_size : int, default = 7
        Length of the 1D convolution window.
    avg_pool_size : int, default = 3
        Size of the average pooling window.
    filter_sizes : array-like of int, shape = (n_conv_layers), default = [6, 12]
        Number of filters per conv layer.
    use_bias : bool, default = True
        Whether to use bias in output layer.
    padding : string, default = "auto"
        Controls padding logic for the convolutional layers,
        i.e. whether ``'valid'`` and ``'same'`` are passed to the ``Conv1D`` layer.
        - "auto": as per original implementation, ``"same"`` is passed if
          ``input_shape[0] < 60`` in the input layer, and ``"valid"`` otherwise.
        - "valid", "same", and other values are passed directly to ``Conv1D``
    activation : str or None, default = None
        Activation for the output layer.
    activation_hidden : str, default = "sigmoid"
        Activation for hidden conv layers: "sigmoid" or "relu".
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
        "authors": ["hfawaz", "AurumnPegasus", "achieveordie", "noxthot"],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
        "tests:vm": True,
    }

    def __init__(
        self,
        num_epochs=2000,
        n_conv_layers=2,
        batch_size=16,
        kernel_size=7,
        avg_pool_size=3,
        filter_sizes=[6, 12],
        use_bias=True,
        padding="auto",
        activation=None,
        activation_hidden="sigmoid",
        optimizer="Adam",
        optimizer_kwargs=None,
        criterion="MSELoss",
        criterion_kwargs=None,
        callbacks="ReduceLROnPlateau",
        callback_kwargs=None,
        lr=0.01,
        verbose=False,
        random_state=None,
    ):
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
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
        self.random_state = random_state

        if len(filter_sizes) != n_conv_layers:
            raise ValueError(
                f"Length of filter_sizes ({len(filter_sizes)}) must match "
                f"n_conv_layers ({n_conv_layers}) in CNNRegressorTorch."
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
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}."
            )
        n_dims = X.shape[1]
        series_length = X.shape[2]
        input_shape = (n_dims, series_length)

        return CNNNetworkTorch(
            input_shape=input_shape,
            num_classes=1,
            kernel_size=self.kernel_size,
            avg_pool_size=self.avg_pool_size,
            n_conv_layers=self.n_conv_layers,
            filter_sizes=self.filter_sizes,
            activation_hidden=self.activation_hidden,
            use_bias=self.use_bias,
            activation=self.activation,
            padding=self.padding,
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
        }
        params2 = {
            "num_epochs": 12,
            "batch_size": 6,
            "kernel_size": 2,
            "n_conv_layers": 1,
        }
        params3 = {
            "num_epochs": 8,
            "batch_size": 4,
            "kernel_size": 5,
            "padding": "same",
            "activation_hidden": "relu",
        }
        return [params1, params2, params3]
