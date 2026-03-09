"""Time Series Attentional Prototype Network (TapNet) classifier in PyTorch."""

__authors__ = ["dakshhhhh16"]
__all__ = ["TapNetClassifierTorch"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.tapnet import TapNetNetworkTorch


class TapNetClassifierTorch(BaseDeepClassifierPytorch):
    """Time series attentional prototype network (TapNet) in PyTorch, as in [1]_.

    Parameters
    filter_sizes : tuple of int, default = (256, 256, 128)
        Number of convolutional filters in each convolutional block.
    kernel_size : tuple of int, default = (8, 5, 3)
        Size of the convolutional kernels.
    layers : tuple of int, default = (500, 300)
        Size of dense layers in the mapping section.
    dropout : float, default = 0.5
        Dropout rate in the CNN branch, in the range [0, 1).
    lstm_dropout : float, default = 0.8
        Dropout rate for the LSTM branch, in the range [0, 1).
    dilation : int, default = 1
        Dilation value for convolutional layers.
    padding : str, default = "same"
        Type of padding for convolution layers.
    use_rp : bool, default = True
        Whether to use random projections (channel subsampling).
    rp_params : tuple of int, default = (-1, 3)
        Parameters for random permutation: (rp_group, rp_dim).
        If rp_group < 0, it is set to 3 and rp_dim to floor(n_channels * 2/3).
    activation_hidden : str, default = "leaky_relu"
        Activation function used in the hidden layers.
    use_att : bool, default = True
        Whether to use self-attention.
    use_lstm : bool, default = True
        Whether to use an LSTM branch.
    use_cnn : bool, default = True
        Whether to use a CNN branch.
    use_bias : bool, default = True
        Whether to use bias in the output dense layer.
    num_epochs : int, default = 2000
        Number of epochs to train the model.
    batch_size : int, default = 16
        Number of samples per mini-batch.
    activation : str or None, default = None
        Activation function for the output layer.
        If None, no activation is applied (raw logits for CrossEntropyLoss).
    criterion : str or None or Callable, default = None
        Loss function. If None, CrossEntropyLoss is used.
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments for the loss function.
    optimizer : str or None or Callable, default = None
        Optimizer. If None, Adam is used.
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments for the optimizer.
    callbacks : None or str or tuple of str, default = None
        Learning rate scheduler(s).
    callback_kwargs : dict or None, default = None
        Keyword arguments for the learning rate scheduler(s).
    lr : float, default = 0.01
        Learning rate for the optimizer.
    verbose : bool, default = False
        Whether to output extra information during training.
    random_state : int or None, default = None
        Seed for reproducibility.

    Attributes
    n_classes_ : int
        Number of classes, extracted from the data.

    Examples

    >>> from sktime.classification.deep_learning.tapnet import TapNetClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> tapnet = TapNetClassifierTorch(num_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> tapnet.fit(X_train, y_train)  # doctest: +SKIP
    TapNetClassifierTorch(...)
    """

    _tags = {
        # packaging info
        "authors": ["dakshhhhh16"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "TapNetClassifierTorch",
        # model specific
        filter_sizes: tuple = (256, 256, 128),
        kernel_size: tuple = (8, 5, 3),
        layers: tuple = (500, 300),
        dropout: float = 0.5,
        lstm_dropout: float = 0.8,
        dilation: int = 1,
        padding: str = "same",
        use_rp: bool = True,
        rp_params: tuple = (-1, 3),
        activation_hidden: str = "leaky_relu",
        use_att: bool = True,
        use_lstm: bool = True,
        use_cnn: bool = True,
        use_bias: bool = True,
        # base classifier specific
        num_epochs: int = 2000,
        batch_size: int = 16,
        activation: str | None | Callable = None,
        criterion: str | None | Callable = None,
        criterion_kwargs: dict | None = None,
        optimizer: str | None | Callable = None,
        optimizer_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = None,
        callback_kwargs: dict | None = None,
        lr: float = 0.01,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        # model-specific params
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.layers = layers
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.dilation = dilation
        self.padding = padding
        self.use_rp = use_rp
        self.rp_params = rp_params
        self.activation_hidden = activation_hidden
        self.use_att = use_att
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.use_bias = use_bias

        # base classifier params
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.activation = activation
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.callbacks = callbacks
        self.callback_kwargs = callback_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # to be inferred from data
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
        X : np.ndarray of shape (n_instances, n_dims, series_length)
            Input training data.
        y : np.ndarray of shape (n_instances,)
            Target labels.

        Returns
        network : TapNetNetworkTorch
            The constructed TapNet network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )

        self.num_classes = len(np.unique(y))
        _, n_channels, series_length = X.shape

        return TapNetNetworkTorch(
            n_channels=n_channels,
            series_length=series_length,
            activation=self.activation_hidden,
            kernel_size=self.kernel_size,
            layers=self.layers,
            filter_sizes=self.filter_sizes,
            random_state=self.random_state,
            rp_params=self.rp_params,
            dropout=self.dropout,
            lstm_dropout=self.lstm_dropout,
            dilation=self.dilation,
            padding=self.padding,
            use_rp=self.use_rp,
            use_att=self.use_att,
            use_lstm=self.use_lstm,
            use_cnn=self.use_cnn,
            num_classes=self.num_classes,
            activation_output=self._validated_activation,
            use_bias=self.use_bias,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters

        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns

        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
        """
        param1 = {
            "num_epochs": 20,
            "batch_size": 4,
            "use_lstm": False,
            "use_att": False,
            "filter_sizes": (16, 16, 16),
            "dilation": 2,
            "layers": (32, 16),
        }
        param2 = {
            "num_epochs": 20,
            "use_cnn": False,
            "layers": (25, 25),
        }
        param3 = {
            "num_epochs": 10,
            "batch_size": 4,
            "padding": "valid",
            "filter_sizes": (16, 16, 16),
            "kernel_size": (3, 3, 1),
            "layers": (25, 50),
            "use_rp": False,
        }
        return [param1, param2, param3]
