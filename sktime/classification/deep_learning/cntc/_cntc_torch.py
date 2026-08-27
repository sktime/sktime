"""Contextual Time-series Neural Classifier for TSC.

Implemented in torch backend.
"""

__authors__ = ["fnhirwa", "srupat"]
__all__ = ["CNTCClassifierTorch"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.cntc import CNTCNetworkTorch


class CNTCClassifierTorch(BaseDeepClassifierPytorch):
    """Contextual Time-series Neural Classifier for TSC, implemented in PyTorch.

    CNTC combines a Contextual Convolutional Neural Network (CCNN) and a
    Contextual Long Short-Term Memory network (CLSTM) as parallel feature
    extractors, concatenates their outputs per time step, refines them with
    self-attention and classifies with a multilayer perceptron, following [1]_.

    The four stages are:

    - **Feature extraction.** The CCNN arm stacks contextual convolutional
      layers (recurrent convolutions, equation 1 of [1]_) followed by standard
      convolutional layers. The CLSTM arm stacks contextual LSTM layers whose
      gates receive sliding-window means of the input as contextual features
      (equation 4 of [1]_). Both arms see the same input.
    - **Concatenation.** The two arms are concatenated along the feature axis,
      per time step, giving ``c_k = concat(mu_k, h_k)`` (equation 5 of [1]_).
    - **Attention.** Pooling downsamples the merged sequence, then sequential
      self-attention reweights it (equations 6 to 8 of [1]_).
    - **Multilayer perceptron.** Fully connected layers with dropout, followed
      by the output layer.

    Parameters
    ----------
    context_filter_sizes : tuple of int, default = (16,)
        Number of filters in each contextual convolutional layer of the CCNN
        arm. The length of the tuple sets the number of such layers; any length
        >= 1 is allowed. [1]_ uses a single layer with 8, 16, 32 or 64 filters.
    context_kernel_sizes : tuple of int, default = (3,)
        Length of the 1D convolution window of each contextual convolutional
        layer. Must have the same length as ``context_filter_sizes``. Combined
        with ``context_steps``, a kernel of size ``n`` gives an effective
        receptive field of ``(n - 1) * context_steps + 1``.
    context_steps : int, default = 3
        Number of recurrent iterations ``K`` performed inside every contextual
        convolutional layer. ``context_steps=1`` reduces the layer to an
        ordinary convolution.
    context_dropout : float or tuple of float, default = 0.8
        Dropout rate applied after each contextual convolutional layer. A float
        applies the same rate to all of them, a tuple sets them individually and
        must have the same length as ``context_filter_sizes``.
    conv_filter_sizes : tuple of int, default = (8,)
        Number of filters in each standard convolutional layer, applied after
        the contextual convolutional layers. The length of the tuple sets the
        number of such layers; ``()`` disables them entirely. [1]_ uses a single
        layer with 8, 16 or 32 filters.
    conv_kernel_sizes : tuple of int, default = (3,)
        Length of the 1D convolution window of each standard convolutional
        layer. Must have the same length as ``conv_filter_sizes``.
    conv_dropout : float or tuple of float, default = 0.8
        Dropout rate applied after each standard convolutional layer. A float
        applies the same rate to all of them, a tuple sets them individually and
        must have the same length as ``conv_filter_sizes``.
    lstm_units : tuple of int, default = (8,)
        Number of cells in each contextual LSTM layer of the CLSTM arm. The
        length of the tuple sets the number of stacked layers; any length >= 1
        is allowed. [1]_ uses a single layer with 8, 16, 32 or 64 cells.
    context_window : int, default = 3
        Size of the sliding window used to build the contextual features of the
        CLSTM arm. At each time step the context vector holds the
        ``context_window`` most recent window means, so it has dimension
        ``context_window * n_dims``.
    lstm_dropout : float or tuple of float, default = 0.8
        Dropout rate applied after each contextual LSTM layer. A float applies
        the same rate to all of them, a tuple sets them individually and must
        have the same length as ``lstm_units``. [1]_ uses 0.8.
    pool_size : int, default = 2
        Size and stride of the pooling window applied to the merged sequence
        before attention. Values above 1 downsample the time axis.
        Clipped to the series length.
    pool_type : str or None, default = "max"
        Pooling to apply before attention. One of ``"max"``, ``"avg"``,
        ``"both"`` (max and average pooling concatenated along the feature
        axis), or ``None`` to disable pooling.
    pool_dropout : float, default = 0.6
        Dropout rate applied after pooling.
    attention_width : int or None, default = 10
        Width of the local attention window. ``None`` lets every time step
        attend to every other time step. [1]_ uses 8 or 10.
    attention_units : int, default = 32
        Hidden dimension of the additive attention scorer. Unused when
        ``attention_type="multiplicative"``.
    attention_type : str, default = "additive"
        Attention scoring function, either ``"additive"`` or
        ``"multiplicative"``. [1]_ scores alignments with a feedforward network,
        which corresponds to ``"additive"``.
    attention_activation : str, callable, torch.nn.Module or None, default = None
        Non-linearity applied to the attention logits before normalisation.
        ``None`` normalises the raw alignment scores with a softmax, as in
        equation (7) of [1]_.
    attention_dropout : float, default = 0.5
        Dropout rate applied after attention. [1]_ uses 0.5.
    dense_layers : tuple of int, default = (64, 64)
        Number of units in each dense layer of the multilayer perceptron. The
        length of the tuple sets the number of such layers; ``()`` connects the
        attention output straight to the output layer. [1]_ uses two layers of
        64 units.
    dense_dropout : float or tuple of float, default = (0.5, 0.8)
        Dropout rate applied after each dense layer. A float applies the same
        rate to all of them, a tuple sets them individually and must have the
        same length as ``dense_layers``. [1]_ uses 0.5 and 0.8.
    activation : str, Callable, or None, default=None
        Activation applied to the output layer.

        Permitted values:

        - ``None``: no activation is applied to the output layer and the network
          returns raw outputs (logits). This is typically required when using
          ``CrossEntropyLoss``, which expects logits as input.
        - ``str``: name of a class in ``torch.nn``. Case-sensitive names are
          recommended and must match PyTorch (e.g., ``"ReLU"``, ``"LeakyReLU"``).
          Lowercase aliases for common activations are also accepted
          (e.g., ``"relu"`` is resolved to ``"ReLU"``). The class is instantiated
          with default constructor arguments. Must be a valid ``torch.nn``
          activation; see
          https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        - ``torch.nn.Module``: an instance of a ``torch.nn.Module`` subclass,
          for example ``torch.nn.ReLU()``. Arbitrary callables are not supported.

    activation_hidden : str, Callable, or None, default="ReLU"
        Activation applied to the hidden layers, that is the convolutional
        layers of the CCNN arm and the dense layers of the multilayer
        perceptron.

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

    init_weights : str or None, default = "xavier_uniform"
        The method used to initialize the weights of the convolutional and
        linear layers. Supported values are ``"kaiming_uniform"``,
        ``"kaiming_normal"``, ``"xavier_uniform"``, ``"xavier_normal"``, or
        ``None`` for the default PyTorch initialization. Biases are zeroed
        whenever a method is given.
    num_epochs : int, default = 150
        The number of epochs to train the model.
    batch_size : int, default = 16
        The size of each mini-batch during training. [1]_ uses 16, 32 or 64.
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "Adam"
        The optimizer to use for training the model. List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    criterion : case insensitive str or None or an instance of a loss function
        defined in PyTorch, default = "CrossEntropyLoss"
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
    metrics : None or str or Callable or tuple of str and/or Callable, default = None
        Metrics to compute during training. If None, no metrics are computed beyond
        the loss. Metrics are computed from torchmetrics library.
        If a string/Callable is passed, it must be one of the metrics defined in
        https://lightning.ai/docs/torchmetrics/stable/
        Examples: "Accuracy", "F1Score", "Precision", "Recall"
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Amadu Fullah Kamara, Enhong Chen, Qi Liu, Zhen Pan, Combin-
       ing Contextual Neural Networks for Time Series Classification,
       Neurocomputing (2019), doi:
       https://doi.org/10.1016/j.neucom.2019.10.113

    Examples
    --------
    >>> from sktime.classification.deep_learning.cntc import CNTCClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> clf = CNTCClassifierTorch(num_epochs=5, batch_size=4)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    CNTCClassifierTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": __authors__,
        "maintainers": ["fnhirwa", "srupat"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "CNTCClassifierTorch",
        # model specific
        context_filter_sizes: tuple[int, ...] = (16,),
        context_kernel_sizes: tuple[int, ...] = (3,),
        context_steps: int = 3,
        context_dropout: float | tuple[float, ...] = 0.8,
        conv_filter_sizes: tuple[int, ...] = (8,),
        conv_kernel_sizes: tuple[int, ...] = (3,),
        conv_dropout: float | tuple[float, ...] = 0.8,
        lstm_units: tuple[int, ...] = (8,),
        context_window: int = 3,
        lstm_dropout: float | tuple[float, ...] = 0.8,
        pool_size: int = 2,
        pool_type: str | None = "max",
        pool_dropout: float = 0.6,
        attention_width: int | None = 10,
        attention_units: int = 32,
        attention_type: str = "additive",
        attention_activation: str | Callable | None = None,
        attention_dropout: float = 0.5,
        dense_layers: tuple[int, ...] = (64, 64),
        dense_dropout: float | tuple[float, ...] = (0.5, 0.8),
        activation: str | Callable | None = None,
        activation_hidden: str | Callable | None = "ReLU",
        init_weights: str | None = "xavier_uniform",
        # base classifier specific
        num_epochs: int = 150,
        batch_size: int = 16,
        optimizer: str | None | Callable = "Adam",
        criterion: str | None | Callable = "CrossEntropyLoss",
        callbacks: None | str | tuple[str, ...] = "ReduceLROnPlateau",
        optimizer_kwargs: dict | None = None,
        criterion_kwargs: dict | None = None,
        callback_kwargs: dict | None = None,
        metrics: None | str | Callable | tuple[str | Callable, ...] = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.context_filter_sizes = context_filter_sizes
        self.context_kernel_sizes = context_kernel_sizes
        self.context_steps = context_steps
        self.context_dropout = context_dropout
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_dropout = conv_dropout
        self.lstm_units = lstm_units
        self.context_window = context_window
        self.lstm_dropout = lstm_dropout
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.pool_dropout = pool_dropout
        self.attention_width = attention_width
        self.attention_units = attention_units
        self.attention_type = attention_type
        self.attention_activation = attention_activation
        self.attention_dropout = attention_dropout
        self.dense_layers = dense_layers
        self.dense_dropout = dense_dropout
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.init_weights = init_weights
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion_kwargs = criterion_kwargs
        self.callback_kwargs = callback_kwargs
        self.metrics = metrics
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

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
            metrics=self.metrics,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        # input_size and num_classes are inferred from the data in _build_network
        self.input_size = None
        self.num_classes = None

        super().__post_init__()

    def _build_network(self, X, y):
        """Build the CNTC network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels for the classification task.

        Returns
        -------
        model : CNTCNetworkTorch
            An instance of the CNTCNetworkTorch class initialized with the
            appropriate parameters.
        """
        self.num_classes = len(np.unique(y))
        self.input_size = X.shape
        return CNTCNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            activation=self._callable_activations["activation"],
            activation_hidden=self._callable_activations["activation_hidden"],
            context_filter_sizes=self.context_filter_sizes,
            context_kernel_sizes=self.context_kernel_sizes,
            context_steps=self.context_steps,
            context_dropout=self.context_dropout,
            conv_filter_sizes=self.conv_filter_sizes,
            conv_kernel_sizes=self.conv_kernel_sizes,
            conv_dropout=self.conv_dropout,
            lstm_units=self.lstm_units,
            context_window=self.context_window,
            lstm_dropout=self.lstm_dropout,
            pool_size=self.pool_size,
            pool_type=self.pool_type,
            pool_dropout=self.pool_dropout,
            attention_width=self.attention_width,
            attention_units=self.attention_units,
            attention_type=self.attention_type,
            attention_activation=self.attention_activation,
            attention_dropout=self.attention_dropout,
            dense_layers=self.dense_layers,
            dense_dropout=self.dense_dropout,
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
        # small paper-shaped architecture, trained for a single epoch
        params1 = {
            "context_filter_sizes": (4,),
            "conv_filter_sizes": (4,),
            "lstm_units": (4,),
            "dense_layers": (8, 8),
            "attention_units": 4,
            "num_epochs": 1,
            "batch_size": 2,
            "callbacks": None,
            "verbose": False,
            "random_state": 0,
        }
        # deeper CCNN arm, no standard convolutions, stacked CLSTM layers
        params2 = {
            "context_filter_sizes": (4, 4),
            "context_kernel_sizes": (3, 1),
            "context_steps": 2,
            "context_dropout": (0.5, 0.2),
            "conv_filter_sizes": (),
            "conv_kernel_sizes": (),
            "lstm_units": (4, 2),
            "context_window": 5,
            "lstm_dropout": 0.3,
            "pool_type": "avg",
            "pool_size": 3,
            "dense_layers": (8,),
            "dense_dropout": 0.4,
            "num_epochs": 1,
            "batch_size": 2,
            "optimizer": "RMSprop",
            "callbacks": None,
            "verbose": False,
            "random_state": 0,
        }
        # no pooling, no dense layers, multiplicative attention, no weight init
        params3 = {
            "context_filter_sizes": (2,),
            "context_kernel_sizes": (1,),
            "context_steps": 1,
            "conv_filter_sizes": (2,),
            "conv_kernel_sizes": (1,),
            "lstm_units": (2,),
            "pool_type": None,
            "attention_type": "multiplicative",
            "attention_activation": "sigmoid",
            "attention_width": None,
            "dense_layers": (),
            "init_weights": None,
            "activation_hidden": "Tanh",
            "num_epochs": 1,
            "batch_size": 2,
            "optimizer": "SGD",
            "optimizer_kwargs": {"momentum": 0.9},
            "callbacks": None,
            "verbose": False,
            "random_state": 0,
        }
        # both poolings concatenated, scheduler enabled, zero dropout
        params4 = {
            "context_filter_sizes": (4,),
            "conv_filter_sizes": (4,),
            "lstm_units": (4,),
            "context_dropout": 0.0,
            "conv_dropout": 0.0,
            "lstm_dropout": 0.0,
            "pool_dropout": 0.0,
            "attention_dropout": 0.0,
            "dense_layers": (8, 8),
            "dense_dropout": 0.0,
            "attention_units": 4,
            "pool_type": "both",
            "num_epochs": 1,
            "batch_size": 2,
            "callbacks": "ReduceLROnPlateau",
            "callback_kwargs": {"factor": 0.7, "patience": 50},
            "verbose": False,
            "random_state": 0,
        }
        return [params1, params2, params3, params4]
