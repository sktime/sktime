"""TapNet Neural Network for classification in PyTorch."""

__authors__ = ["srupat"]
__all__ = ["TapNetClassifierTorch"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.tapnet import TapNetNetworkTorch


class TapNetClassifierTorch(BaseDeepClassifierPytorch):
    """TapNet classifier in PyTorch.

    Parameters
    ----------
    filter_sizes : tuple of int, default = (256, 256, 128)
        Number of convolutional filters in each conv block.
        If ``use_rp`` is True, the first conv layer is group-specific and all
        subsequent conv layers share parameters across groups.
    kernel_size : tuple of int, default = (8, 5, 3)
        Specifying the length of the 1D convolution window.
    layers : tuple of int, default = (500, 300)
        Size of dense layers in the mapping section.
    dropout : float, default = 0.5
        Dropout rate for the convolutional layers.
    lstm_dropout : float, default = 0.8
        Dropout rate for the LSTM layer.
    dilation : int, default = 1
        Dilation value.
    activation : str or None, default = None
        Activation function to use in the output layer.
    activation_hidden : str, default = "leaky_relu"
        Activation function to use in the hidden layers.
    use_rp : bool, default = True
        Whether to use random projections.
    rp_group : int, default = 3
        Number of random permutation groups g for random dimension permutation (RDP).
        Must be a positive integer.
    rp_alpha : float, default = 2.0
        Scale factor alpha used to compute the RDP group size:
        rp_dim = floor(n_dims * rp_alpha / rp_group).
        If rp_dim becomes 0, RDP is disabled with a warning (RDP requires
        multivariate inputs).
        Must be positive.
    use_att : bool, default = True
        Whether to use self attention.
    use_lstm : bool, default = True
        Whether to use an LSTM layer.
    use_cnn : bool, default = True
        Whether to use a CNN layer.
    padding : str, default = "same"
        Type of padding for convolution layers.
    init_weights : bool, default = True
        Whether to apply custom initialization.
    fc_dropout : float, default = 0.0
        Dropout rate before the output layer.
    num_epochs : int, default = 100
        The number of epochs to train the model.
    batch_size : int, default = 1
        The size of each mini-batch during training.
    optimizer : str or None or an instance of optimizers
        defined in torch.optim, default = "RMSprop"
        The optimizer to use for training the model.
        List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    criterion : str or None or an instance of a loss function
        defined in PyTorch, default = "CrossEntropyLoss"
        The loss function to be used in training the neural network.
        List of available loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions
    callbacks : None or str or a tuple of str, default = "ReduceLROnPlateau"
        Learning rate schedulers applied during training.
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
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    Examples
    --------
    >>> from sktime.classification.deep_learning.tapnet import TapNetClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = TapNetClassifierTorch(num_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    TapNetClassifierTorch(...)
    """

    _tags = {
        "authors": ["srupat"],
        "maintainers": ["srupat"],
        "python_version": ">=3.10, <3.15",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
        "capability:multivariate": True,
    }

    def __init__(
        self: "TapNetClassifierTorch",
        # model specific
        filter_sizes: tuple[int, ...] = (256, 256, 128),
        kernel_size: tuple[int, ...] = (8, 5, 3),
        layers: tuple[int, ...] = (500, 300),
        dropout: float = 0.5,
        lstm_dropout: float = 0.8,
        dilation: int = 1,
        activation: str | None | Callable = None,
        activation_hidden: str = "leaky_relu",
        use_rp: bool = True,
        rp_group: int = 3,
        rp_alpha: float = 2.0,
        use_att: bool = True,
        use_lstm: bool = True,
        use_cnn: bool = True,
        padding: str = "same",
        init_weights: bool = True,
        fc_dropout: float = 0.0,
        # base classifier specific
        num_epochs: int = 100,
        batch_size: int = 1,
        optimizer: str | None | Callable = "RMSprop",
        criterion: str | None | Callable = "CrossEntropyLoss",
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        optimizer_kwargs: dict | None = None,
        criterion_kwargs: dict | None = None,
        callback_kwargs: dict | None = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.layers = layers
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.dilation = dilation
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.use_rp = use_rp
        self.rp_group = rp_group
        self.rp_alpha = rp_alpha
        self.use_att = use_att
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.padding = padding
        self.init_weights = init_weights
        self.fc_dropout = fc_dropout
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

        # input_size and num_classes inferred from the data and will be
        # set in build_network
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
        """Build the TapNet network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels corresponding to the input data.

        Returns
        -------
        model : TapNetNetworkTorch
             An instance of the TapNetNetworkTorch class initialized with the
             appropriate parameters.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )
        self.num_classes = len(np.unique(y))
        self.input_size = X.shape
        return TapNetNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            activation=self._validated_activation,
            activation_hidden=self.activation_hidden,
            kernel_size=self.kernel_size,
            layers=self.layers,
            filter_sizes=self.filter_sizes,
            dropout=self.dropout,
            lstm_dropout=self.lstm_dropout,
            dilation=self.dilation,
            padding=self.padding,
            use_rp=self.use_rp,
            rp_group=self.rp_group,
            rp_alpha=self.rp_alpha,
            use_att=self.use_att,
            use_lstm=self.use_lstm,
            use_cnn=self.use_cnn,
            random_state=self.random_state,
            init_weights=self.init_weights,
            fc_dropout=self.fc_dropout,
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
            "num_epochs": 20,
            "batch_size": 4,
            "use_lstm": False,
            "use_att": False,
            "filter_sizes": (16, 16, 16),
            "dilation": 2,
            "layers": (32, 16),
        }
        params3 = {
            "num_epochs": 20,
            "use_cnn": False,
            "layers": (25, 25),
        }
        # no attention + no rp
        params4 = {
            "num_epochs": 2,
            "use_att": False,
            "use_rp": False,
            "filter_sizes": (8, 8, 8),
            "layers": (16, 8),
        }
        # cnn + lstm with rp
        params5 = {
            "num_epochs": 2,
            "use_rp": True,
            "rp_group": 2,
            "rp_alpha": 1.0,
            "filter_sizes": (8, 8, 8),
            "layers": (16, 8),
        }
        return [params1, params2, params3, params4, params5]
