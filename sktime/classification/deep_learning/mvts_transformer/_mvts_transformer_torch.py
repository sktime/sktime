"""Multivariate time series transformer (PyTorch) classifier."""

__authors__ = ["gzerveas", "geetu040"]
__all__ = ["MVTSTransformerClassifier"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.mvts_transformer import MVTSTransformerNetworkTorch
from sktime.utils.dependencies import _safe_import


class MVTSTransformerClassifier(BaseDeepClassifierPytorch):
    """Multivariate Time Series Transformer for Classification, as described in [1]_.

    This classifier has been wrapped around the official pytorch implementation of
    Transformer from [2]_, provided by the authors of the paper [1]_.

    Parameters
    ----------
    d_model : int, optional (default=256)
        The number of expected features in the input (i.e., the dimension of the model).
    n_heads : int, optional (default=4)
        The number of heads in the multihead attention mechanism.
    num_layers : int, optional (default=4)
        The number of layers (or blocks) in the transformer encoder.
    dim_feedforward : int, optional (default=128)
        The dimension of the feedforward network model.
    dropout : float, optional (default=0.1)
        The dropout rate to apply.
    pos_encoding : str, optional (default="fixed")
        The type of positional encoding to use. Options: ["fixed", "learnable"].
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

    activation_hidden : str, Callable, or None, default="relu"
        Activation applied to the hidden layers, i.e., inside the transformer
        encoder layers and after the encoder stack.

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

    norm : str, optional (default="BatchNorm")
        The type of normalization to use in the encoder layers.
        Options: ["BatchNorm", "LayerNorm"].
    freeze : bool, optional (default=False)
        If True, dropout is disabled in the positional encoding and in the
        transformer encoder layers, as in the reference implementation.
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
        The learning rate for the optimizer.
    verbose : bool, optional (default=True)
        If True, prints progress messages during training.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.classification.deep_learning import MVTSTransformerClassifier
    >>>
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, _ = load_unit_test(split="test")
    >>>
    >>> model = MVTSTransformerClassifier()
    >>> model.fit(X_train, y_train)  # doctest: +SKIP
    >>> preds = model.predict(X_test)  # doctest: +SKIP

    References
    ----------
    .. [1] George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha Bhamidipaty,
    and Carsten Eickhoff. 2021. A Transformer-based Framework
    for Multivariate Time Series Representation Learning.
    In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery
    & Data Mining (KDD '21). Association for Computing Machinery, New York, NY, USA,
    2114-2124. https://doi.org/10.1145/3447548.3467401.
    .. [2] https://github.com/gzerveas/mvts_transformer
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["gzerveas", "geetu040"],
        # gzerveas for original code in research repository
        "maintainers": ["geetu040"],
        "python_dependencies": ["torch"],
        # estimator type
        # --------------
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # CI and testing
        # --------------
        "tests:vm": True,
        "tests:libs": [
            "sktime.networks.mvts_transformer._mvts_transformer_torch",
            "sktime.networks.mvts_transformer._encoder_layer",
            "sktime.networks.mvts_transformer._positional_encoding",
            "sktime.libs._torch_positional_encoding",
        ],
    }

    def __init__(
        self: "MVTSTransformerClassifier",
        # model specific
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pos_encoding: str = "fixed",
        activation: str | None | Callable = None,
        activation_hidden: str | None | Callable = "relu",
        norm: str = "BatchNorm",
        freeze: bool = False,
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
        verbose: bool = True,
        random_state: int | None = None,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.pos_encoding = pos_encoding
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.norm = norm
        self.freeze = freeze
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
        # feat_dim, max_len and num_classes are inferred from the data
        # and will be set in _build_network
        self.feat_dim = None
        self.max_len = None
        self.num_classes = None

        super().__post_init__()

    def _build_network(self, X, y):
        """Build the multivariate time series transformer network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels corresponding to the input data.

        Returns
        -------
        model : MVTSTransformerNetworkTorch
            An instance of the MVTSTransformerNetworkTorch class initialized with
            the appropriate parameters.
        """
        # n_instances, n_dims, n_timestamps
        _, self.feat_dim, self.max_len = X.shape

        self.num_classes = len(np.unique(y))

        return MVTSTransformerNetworkTorch(
            feat_dim=self.feat_dim,
            max_len=self.max_len,
            num_classes=self.num_classes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            pos_encoding=self.pos_encoding,
            activation=self._callable_activations["activation"],
            activation_hidden=self._callable_activations["activation_hidden"],
            norm=self.norm,
            freeze=self.freeze,
        )

    def _build_dataloader(self, X, y=None):
        """Build a dataloader that also emits the padding masks of the network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray, optional
            Target labels. If None, the dataloader is for inference.

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader over ``PytorchDataset``.
        """
        DataLoader = _safe_import("torch.utils.data.DataLoader")
        dataset = PytorchDataset(X, y)
        return DataLoader(dataset, self.batch_size)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {
                "d_model": 16,
                "n_heads": 1,
                "num_layers": 1,
                "dim_feedforward": 8,
                "dropout": 0,
                "pos_encoding": "fixed",
                "activation_hidden": "relu",
                "norm": "BatchNorm",
                "freeze": False,
                "num_epochs": 1,
                "verbose": False,
                "random_state": 0,
            },
            {
                "d_model": 16,
                "n_heads": 1,
                "num_layers": 1,
                "dim_feedforward": 8,
                "dropout": 0,
                "pos_encoding": "learnable",
                "activation_hidden": "gelu",
                "norm": "LayerNorm",
                "freeze": True,
                "num_epochs": 1,
                "verbose": False,
                "random_state": 0,
            },
            {
                "d_model": 16,
                "n_heads": 2,
                "num_layers": 1,
                "dim_feedforward": 8,
                "dropout": 0,
                "pos_encoding": "fixed",
                "activation_hidden": "gelu",
                "norm": "BatchNorm",
                "freeze": False,
                "num_epochs": 1,
                "verbose": False,
                "random_state": 0,
                "optimizer": "AdamW",
                "callbacks": "ReduceLROnPlateau",
            },
        ]
        return params


Dataset = _safe_import("torch.utils.data.Dataset")


class PytorchDataset(Dataset):
    """Dataset for the multivariate time series transformer classifier.

    In addition to the series itself, this dataset emits the boolean padding
    masks that the transformer encoder expects.
    """

    def __init__(self, X, y=None):
        # X.shape = (batch_size, n_dims, n_timestamps)
        X = np.transpose(X, (0, 2, 1))
        # X.shape = (batch_size, n_timestamps, n_dims)

        self.X = X
        self.y = y

    def __len__(self):
        """Get length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Get item at index."""
        torchTensor = _safe_import("torch.tensor")
        torchOnes = _safe_import("torch.ones")
        torchFloat = _safe_import("torch.float")
        torchLong = _safe_import("torch.long")
        torchBool = _safe_import("torch.bool")

        x = self.X[i]
        x = torchTensor(x, dtype=torchFloat)
        padding_masks = torchOnes(x.shape[:-1], dtype=torchBool)

        inputs = {
            "X": x,
            "padding_masks": padding_masks,
        }

        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torchTensor(y, dtype=torchLong)
        return inputs, y
