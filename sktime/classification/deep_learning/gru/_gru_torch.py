"""Gated Recurrent Unit (GRU) classifier in PyTorch."""

__authors__ = ["fnhirwa"]
__all__ = ["GRUClassifier"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.gru import GRUNetworkTorch


class GRUClassifier(BaseDeepClassifierPytorch):
    """Gated Recurrent Unit (GRU) for time series classification.

    This classifier has been wrapped around implementations from [1]_, [2]_ and [3]_.

    Parameters
    ----------
    hidden_dim : int, optional (default=256)
        Number of features in the hidden state.
    n_layers : int, optional (default=4)
        Number of recurrent layers.
    bias : bool, optional (default=True)
        If False, then the layer does not use bias weights.
    init_weights : bool, optional (default=True)
        If True, then the weights are initialized with a TensorFlow like
        initialization.
    dropout : float, optional (default=0.0)
        Dropout rate applied between the recurrent layers.
    fc_dropout : float, optional (default=0.0)
        Dropout rate applied before the fully connected layer.
    bidirectional : bool, optional (default=False)
        If True, then the GRU is bidirectional.
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

    num_epochs : int, optional (default=10)
        The number of epochs to train the model.
    batch_size : int, optional (default=8)
        The size of each mini-batch during training.
    criterion : case insensitive str or an instance of a loss function
        defined in PyTorch, optional (default=None)
        The loss function to be used in training the neural network.
        If None, CrossEntropyLoss is used.
        List of available loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions
    criterion_kwargs : dict or None, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    optimizer : case insensitive str, or a class or instance of an optimizer
        defined in PyTorch, optional (default=None)
        The optimizer to use for training the model. If None, Adam is used.
        List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict or None, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
    callbacks : None or str or a tuple of str, optional (default=None)
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
    callback_kwargs : dict or None, optional (default=None)
        The keyword arguments to be passed to the callbacks.
    metrics : None or str or Callable or tuple of str and/or Callable,
        optional (default=None)
        Metrics to compute during training. If None, no metrics are computed beyond
        the loss. Metrics are computed from the torchmetrics library.
        If a string/Callable is passed, it must be one of the metrics defined in
        https://lightning.ai/docs/torchmetrics/stable/
        Examples: "Accuracy", "F1Score", "Precision", "Recall"
    lr : float, optional (default=0.001)
        The learning rate to use for the optimizer.
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    random_state : int or None, optional (default=None)
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Cho, Kyunghyun, et al. "Learning phrase representations
        using RNN encoder-decoder for statistical machine translation."
        arXiv preprint arXiv:1406.1078 (2014).
    .. [2] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio.
        Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
        arXiv preprint arXiv:1412.3555 (2014).
    .. [3] https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

    Examples
    --------
    >>> from sktime.classification.deep_learning.gru import GRUClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> clf = GRUClassifier(num_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    GRUClassifier(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fnhirwa"],
        "maintainers": ["fnhirwa", "srupat"],
        "python_dependencies": "torch",
        # estimator type
        # --------------
        # remaining tags handled by BaseDeepClassifierPytorch
        "property:randomness": "stochastic",
        "capability:random_state": True,
        # CI and testing
        # --------------
        "tests:vm": True,
        "tests:libs": ["sktime.networks.gru._gru_torch"],
    }

    # the network has no hidden layer activation, only an output layer one
    _instantiate_activation_vars = ("activation",)

    def __init__(
        self: "GRUClassifier",
        # model specific
        hidden_dim: int = 256,
        n_layers: int = 4,
        bias: bool = True,
        init_weights: bool = True,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        bidirectional: bool = False,
        activation: str | None | Callable = None,
        # base classifier specific
        num_epochs: int = 10,
        batch_size: int = 8,
        criterion: str | None | Callable = None,
        criterion_kwargs: dict | None = None,
        optimizer: str | None | Callable = None,
        optimizer_kwargs: dict | None = None,
        callbacks: None | str | tuple[str, ...] = None,
        callback_kwargs: dict | None = None,
        metrics: None | str | Callable | tuple[str | Callable, ...] = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bias = bias
        self.init_weights = init_weights
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.bidirectional = bidirectional
        self.activation = activation
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.callbacks = callbacks
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
        # input_size and num_classes are inferred from the data
        # and will be set in _build_network
        self.input_size = None
        self.num_classes = None

        super().__post_init__()

    def _build_network(self, X, y):
        """Build the GRU network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels corresponding to the input data.

        Returns
        -------
        model : GRUNetworkTorch
            An instance of the GRUNetworkTorch class initialized with the
            appropriate parameters.
        """
        self.num_classes = len(np.unique(y))
        _, self.input_size, _ = X.shape

        return GRUNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            # the dataloader of the base class always emits
            # (batch, time, channels)
            batch_first=True,
            bias=self.bias,
            init_weights=self.init_weights,
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
            bidirectional=self.bidirectional,
            activation=self._callable_activations["activation"],
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
        params = [
            {
                "hidden_dim": 256,
                "n_layers": 2,
                "bias": True,
                "init_weights": True,
                "dropout": 0.1,
                "fc_dropout": 0.1,
                "bidirectional": False,
                "num_epochs": 2,
                "optimizer": "Adam",
                "lr": 0.001,
                "verbose": False,
                "random_state": 0,
            },
            {
                "hidden_dim": 64,
                "n_layers": 3,
                "bias": True,
                "init_weights": False,
                "dropout": 0.1,
                "fc_dropout": 0.0,
                "bidirectional": True,
                "num_epochs": 2,
                "optimizer": "Adam",
                "lr": 0.1,
                "verbose": False,
                "random_state": 0,
            },
            {
                "hidden_dim": 16,
                "n_layers": 1,
                "num_epochs": 1,
                "optimizer": "AdamW",
                "criterion": "CrossEntropyLoss",
                "callbacks": "ReduceLROnPlateau",
                "lr": 0.01,
                "verbose": False,
                "random_state": 0,
            },
        ]
        return params
