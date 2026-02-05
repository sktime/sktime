"""Time Convolutional Neural Network (CNN) for regression in PyTorch."""

__authors__ = ["RecreationalMath"]
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
    batch_size : int, default = 16
        Size of each mini-batch.
    kernel_size : int, default = 7
        Length of the 1D convolution window.
    avg_pool_size : int, default = 3
        Size of the average pooling window.
    n_conv_layers : int, default = 2
        Number of convolutional plus average pooling layers.
    filter_sizes : array-like of int, shape = (n_conv_layers), default = None
        Number of filters per conv layer. If None, defaults to [6, 12].
    activation : str or None, default = None
        Activation for the output layer.
    activation_hidden : str, default = "sigmoid"
        Activation for hidden conv layers: "sigmoid" or "relu".
    padding : string, default = "auto"
        Controls padding logic for the convolutional layers,
        i.e. whether ``'valid'`` and ``'same'`` are passed to the ``Conv1D`` layer.
        - "auto": as per original implementation, ``"same"`` is passed if
          ``input_shape[0] < 60`` in the input layer, and ``"valid"`` otherwise.
        - "valid", "same", and other values are passed directly to ``Conv1D``
    optimizer : str or callable, default = "Adam"
        Optimizer to use. Same as TF default (Adam).
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments for the optimizer.
    lr : float, default = 0.01
        Learning rate (TF CNN uses Adam(lr=0.01)).
    criterion : str or callable, default = "MSELoss"
        Loss function (TF uses mean_squared_error).
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments for the criterion.
    callbacks : None or str or tuple of str, default = "ReduceLROnPlateau"
        Learning rate schedulers as callbacks.
    callback_kwargs : dict or None, default = None
        Keyword arguments for callbacks.
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
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>> reg = CNNRegressorTorch(num_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> reg.fit(X_train, y_train)  # doctest: +SKIP
    CNNRegressorTorch(...)
    """

    _tags = {
        "authors": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_fit_idempotent",
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    }

    def __init__(
        self,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        filter_sizes=None,
        activation_hidden="sigmoid",
        activation=None,
        padding="auto",
        num_epochs=2000,
        batch_size=16,
        optimizer="Adam",
        optimizer_kwargs=None,
        lr=0.01,
        criterion="MSELoss",
        criterion_kwargs=None,
        callbacks="ReduceLROnPlateau",
        callback_kwargs=None,
        verbose=False,
        random_state=0,
    ):
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        self.activation_hidden = activation_hidden
        self.activation = activation
        self.padding = padding
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
