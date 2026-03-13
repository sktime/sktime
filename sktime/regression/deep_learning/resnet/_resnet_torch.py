"""Residual Network (ResNet) Regressor in PyTorch."""

__authors__ = ["dakshhhhh16"]
__all__ = ["ResNetRegressorTorch"]

from sktime.networks.resnet import ResNetNetworkTorch
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch


class ResNetRegressorTorch(BaseDeepRegressorTorch):
    """Residual Neural Network Regressor in PyTorch, as described in [1]_.

    Adapted from the TensorFlow implementation from
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    Parameters
    ----------
    n_epochs : int, optional (default=1500)
        The number of epochs to train the model.
    batch_size : int, optional (default=16)
        The number of samples per gradient update.
    n_feature_maps : int, optional (default=64)
        Number of feature maps (filters) in the first residual block.
        Second and third blocks use ``n_feature_maps * 2``.
    criterion : str, optional (default="MSELoss")
        The name of the loss function to be used during training,
        should be supported by PyTorch.
    criterion_kwargs : dict or None, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    activation : str or None, optional (default=None)
        The activation function to apply at the output layer.
        If None, no output activation is applied (linear output).
        List of available activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations
    activation_hidden : str, optional (default="relu")
        Activation function used in the hidden layers.
        List of available activation functions:
        https://pytorch.org/docs/stable/nn.html#non-linear-activations
    use_bias : bool, optional (default=True)
        Whether bias should be included in the output Dense layer.
    optim : str or None, optional (default=None)
        The optimizer to use. If None, Adam is used with lr=0.01.
        List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    optim_kwargs : dict or None, optional (default=None)
        Additional keyword arguments for the optimizer.
    callbacks : None or str or tuple of str, optional (default=None)
        Currently only learning rate schedulers are supported as callbacks.
    callback_kwargs : dict or None, optional (default=None)
        The keyword arguments to be passed to the callbacks.
    lr : float, optional (default=0.01)
        The learning rate for the optimizer.
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    random_state : int or None, optional (default=None)
        Seed for random number generation.

    References
    ----------
    .. [1] Wang et al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.

    Examples
    --------
    >>> from sktime.regression.deep_learning.resnet import ResNetRegressorTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> reg = ResNetRegressorTorch(n_epochs=20)     # doctest: +SKIP
    >>> reg.fit(X_train, y_train)     # doctest: +SKIP
    ResNetRegressorTorch(...)
    """

    _tags = {
        # packaging info
        "authors": ["dakshhhhh16"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "ResNetRegressorTorch",
        n_epochs=1500,
        batch_size=16,
        n_feature_maps=64,
        criterion="MSELoss",
        criterion_kwargs=None,
        activation=None,
        activation_hidden="relu",
        use_bias=True,
        optim=None,
        optim_kwargs=None,
        callbacks=None,
        callback_kwargs=None,
        lr=0.01,
        verbose=False,
        random_state=None,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_feature_maps = n_feature_maps
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.use_bias = use_bias
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # Store original user value
        self.optim = optim
        self.optim_kwargs = optim_kwargs

        # Resolve for base class
        self.optimizer = optim if optim is not None else "Adam"
        self.optimizer_kwargs = optim_kwargs

        super().__init__(
            num_epochs=self.n_epochs,
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
        """Build the ResNet network for regression.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape ``(n_instances, n_dims, series_length)``.

        Returns
        -------
        model : torch.nn.Module
            The constructed ResNet network with single output unit.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape "
                f"(n_instances, n_dims, series_length), "
                f"but got shape {X.shape}."
            )

        n_channels = X.shape[1]
        series_length = X.shape[2]

        return ResNetNetworkTorch(
            n_channels=n_channels,
            series_length=series_length,
            n_feature_maps=self.n_feature_maps,
            activation_hidden=self.activation_hidden,
            activation_output=self.activation,
            use_bias=self.use_bias,
            num_classes=1,
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
            "n_epochs": 6,
            "batch_size": 4,
            "use_bias": False,
        }
        params2 = {
            "n_epochs": 4,
            "batch_size": 6,
            "use_bias": True,
            "n_feature_maps": 32,
        }
        params3 = {
            "n_epochs": 2,
            "batch_size": 4,
            "n_feature_maps": 16,
            "activation_hidden": "relu",
            "lr": 0.005,
            "random_state": 42,
        }
        return [params1, params2, params3]
