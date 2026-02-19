"""Contextual Time-series Neural Classifier for TSC.

Implemented in torch backend.
"""

__authors__ = ["fnhirwa"]
__all__ = ["CNTCClassifierTorch"]


from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.cntc import CNTCNetworkTorch
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
DataLoader = _safe_import("torch.utils.data.DataLoader")
Dataset = _safe_import("torch.utils.data.Dataset")
TensorDataset = _safe_import("torch.utils.data.TensorDataset")


class _CNTCDataset(Dataset):
    """Dataset that returns dict-based inputs for CNTCNetworkTorch.

    Wraps two tensors (x1 and x3) and an optional label tensor, returning
    items as ``({"x1": ..., "x3": ...}, y)`` tuples so that the base-class
    ``_run_epoch`` can unpack them and call ``network(**inputs)`` as
    ``network(x1=..., x3=...)``.
    """

    def __init__(self, X1, X3, y=None):
        self.X1 = X1
        self.X3 = X3
        self.y = y

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        inputs = {"x1": self.X1[idx], "x3": self.X3[idx]}
        if self.y is not None:
            return inputs, self.y[idx]
        return inputs


class CNTCClassifierTorch(BaseDeepClassifierPytorch):
    """Contextual Time-series Neural Classifier for TSC implemented in PyTorch.

    See [1] for details.

    Parameters
    ----------
    kernel_sizes : tuple of int, length n_conv_layers, default=(1, 1)
        Kernel size for each Conv1D layer (first and second conv block).
    rnn_layer : int, default=64
        Hidden size of the SimpleRNN used in the CCNN arm.
    lstm_layer : int, default=8
        Hidden size of the LSTM used in the CLSTM arm.
    evg_pool_size : int, default=1
        Kernel size of the MaxPool1D layer applied after merging all arms.
        The original paper uses 1 (no-op pooling).
    n_conv_layers : int, default=2
        Number of Conv1D blocks in the CCNN arm (must match len(kernel_sizes)
        and len(filter_sizes)).
    filter_sizes : tuple of int, length n_conv_layers, default=(16, 8)
        Number of output filters for each Conv1D block.
    dense_size : int, default=64
        Number of units in each of the two MLP hidden layers.
    hidden_activation : str, default='relu'
        Activation function name for hidden layers (excluding attention).
        Must be a valid attribute of torch.nn.functional, e.g. 'relu', 'tanh'.
    activation_attention : str, default='sigmoid'
        Activation function name applied inside the self-attention score
        computation. Must be a valid attribute of torch.nn.functional.
    dropout : float or tuple of 7 floats, default=(0.8, 0.8, 0.7, 0.8, 0.6, 0.5, 0.8)
        Dropout rate(s) applied in the network. If a single float, the same
        rate is applied everywhere. If a tuple, values correspond to:
        (conv1_dropout, rnn_dropout, conv2_dropout, lstm_dropout,
         pool_dropout, attention_dropout, mlp_dropout)
        where mlp_dropout is shared across both MLP hidden layers.
    init_weights: str, default="xavier_uniform"
        Weight initialization method for all layers. Must be a valid method in
        torch.nn.init, e.g. 'xavier_uniform', 'kaiming_normal', etc.
    random_state : int, default=0
        Seed for reproducible weight initialisation.

    num_epochs : int, default = 100
        The number of epochs to train the model.
    optimizer : case insensitive str or None or an instance of optimizers
        defined in torch.optim, default = "RMSprop"
        The optimizer to use for training the model. List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    optimizer_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the optimizer.
    batch_size : int, default = 1
        The size of each mini-batch during training.
    criterion : case insensitive str or None or an instance of a loss function
        defined in PyTorch, default = "CrossEntropyLoss"
        The loss function to be used in training the neural network.
        List of available loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions
    criterion_kwargs : dict or None, default = None
        Additional keyword arguments to pass to the loss function.
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
    callback_kwargs : dict or None, default = None
        The keyword arguments to be passed to the callbacks.
    lr : float, default = 0.001
        The learning rate to use for the optimizer.
    verbose : bool, default = False
        Whether to print progress information during training.

    References
    ----------
    .. [1] Kamara, A.F., Chen, E., Liu, Q., Pan, Z. (2020).
        Combining contextual neural networks for time series classification.
        Neurocomputing, 384, 57-66.
        https://doi.org/10.1016/j.neucom.2019.10.113

    Examples
    --------
    >>> from sktime.classification.deep_learning.cntc import CNTCClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = CNTCClassifierTorch(
    ...     kernel_sizes=(3, 3),
    ...     rnn_layer=16,
    ...     lstm_layer=8,
    ...     evg_pool_size=2,
    ...     n_conv_layers=2,
    ...     filter_sizes=(16, 8),
    ...     dense_size=32,
    ...     hidden_activation="relu",
    ...     activation_attention="sigmoid",
    ...     dropout=0.5,
    ...     init_weights="xavier_uniform",
    ...     random_state=42,
    ... )
    >>> clf.fit(X_train, y_train, epochs=1)  # doctest: +SKIP
    CNTCClassifierTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": __authors__,
        "maintainers": ["fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self: "CNTCClassifierTorch",
        # model architecture parameters
        kernel_sizes: tuple = (1, 1),
        rnn_layer: int = 64,
        lstm_layer: int = 8,
        evg_pool_size: int = 1,
        n_conv_layers: int = 2,
        filter_sizes: tuple = (16, 8),
        dense_size: int = 64,
        hidden_activation: str = "relu",
        activation_attention: str = "sigmoid",
        dropout: float | tuple = (0.8, 0.8, 0.7, 0.8, 0.6, 0.5, 0.8),
        init_weights: str | None = "xavier_uniform",
        random_state: int = 0,
        # training parameters
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
    ):
        self.kernel_sizes = kernel_sizes
        self.rnn_layer = rnn_layer
        self.lstm_layer = lstm_layer
        self.evg_pool_size = evg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = filter_sizes
        self.dense_size = dense_size
        self.hidden_activation = hidden_activation
        self.activation_attention = activation_attention
        self.dropout = dropout
        self.init_weights = init_weights
        self.random_state = random_state

        # infer the input and output shapes from data
        self.input_size = None
        self.num_classes = None

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=callbacks,
            optimizer_kwargs=optimizer_kwargs,
            criterion_kwargs=criterion_kwargs,
            callback_kwargs=callback_kwargs,
            lr=lr,
            verbose=verbose,
            random_state=random_state,
        )

    @staticmethod
    def _make_x3(X):
        """Compute rolling-mean-augmented input for the CLSTM arm.

        Computes a window-3 rolling mean along the time axis (axis 2 in
        sktime ``[B, n_dims, T]`` format) and returns an array of the
        **same shape** as X.  This mirrors the original Keras
        ``prepare_input`` logic while keeping the channel dimension
        unchanged so it matches the LSTM's ``in_channels``.

        Parameters
        ----------
        X : np.ndarray, shape (B, n_dims, T)
            Raw time-series array.

        Returns
        -------
        X3 : np.ndarray, shape (B, n_dims, T)
        """
        B, n_dims, T = X.shape
        X3 = np.zeros_like(X)
        for t in range(T):
            start = max(0, t - 2)  # window=3
            X3[:, :, t] = X[:, :, start : t + 1].mean(axis=2)
        return X3

    def _build_dataloader(self, X, y=None):
        """Build a DataLoader producing ``({"x1": ..., "x3": ...}, y)`` batches.

        Overrides the base-class implementation so that the base-class
        ``_run_epoch`` can call ``self.network(**inputs)`` as
        ``network(x1=..., x3=...)`` matching ``CNTCNetworkTorch.forward``.

        Both x1 and x3 are transposed from sktime ``[B, n_dims, T]`` to
        ``[B, T, n_dims]`` as required by the network's RNN/LSTM layers.
        """
        training = y is not None
        # X3: rolling-mean augmented version of X with the same shape
        X3 = self._make_x3(X)

        # Transpose [B, n_dims, T] → [B, T, n_dims] to match forward(x1, x3)
        tensor_X1 = torch.from_numpy(X.transpose(0, 2, 1)).float()
        tensor_X3 = torch.from_numpy(X3.transpose(0, 2, 1)).float()

        if y is not None:
            tensor_y = torch.from_numpy(y).long()
            dataset = _CNTCDataset(tensor_X1, tensor_X3, tensor_y)
        else:
            dataset = _CNTCDataset(tensor_X1, tensor_X3)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=training)

    def _get_dataloader(self, X, y=None, training=True):
        """Create a DataLoader for the given data (used by _predict_proba)."""
        X3 = self._make_x3(X)

        # Transpose [B, n_dims, T] → [B, T, n_dims] to match forward(x1, x3)
        tensor_X1 = torch.from_numpy(X.transpose(0, 2, 1)).float()
        tensor_X3 = torch.from_numpy(X3.transpose(0, 2, 1)).float()

        if y is None:
            dataset = TensorDataset(tensor_X1, tensor_X3)
        else:
            tensor_y = torch.from_numpy(y).long()
            dataset = TensorDataset(tensor_X1, tensor_X3, tensor_y)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=training)

    def _training_step(self, batch):
        """Perform a single training step."""
        x1_batch, x3_batch, y_batch = batch
        y_pred = self.network(x1=x1_batch, x3=x3_batch)
        loss = self._criterion(y_pred, y_batch)
        return loss

    def _build_network(self, X, y):
        """Build the CNTC network.

        Parameters
        ----------
        X: np.ndarray
            The input data, containint the timeseries data.
        y: np.ndarray
            The target labels for classification.

        Returns
        -------
        model: torch.nn.Module
            The CNTC network mode (instance of CNTCNetworkTorch).
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )
        # n_instances, n_dims, n_timesteps = X.shape
        self.num_classes = len(np.unique(y))
        # The input size for the network is doubled for the third arm
        self.input_size = X.shape[1]

        return CNTCNetworkTorch(
            in_channels=self.input_size,
            n_classes=self.num_classes,
            kernel_sizes=self.kernel_sizes,
            rnn_layer=self.rnn_layer,
            lstm_layer=self.lstm_layer,
            evg_pool_size=self.evg_pool_size,
            n_conv_layers=self.n_conv_layers,
            filter_sizes=self.filter_sizes,
            dense_size=self.dense_size,
            activation=self.hidden_activation,
            activation_attention=self.activation_attention,
            dropout=self.dropout,
            init_weights=self.init_weights,
            random_state=self.random_state,
        )

    def _predict_proba(self, X, **kwargs):
        """Predict class probabilities for samples in X."""
        self.network.eval()
        dataloader = self._get_dataloader(X, y=None, training=False)

        all_probas = []
        with torch.no_grad():
            for x1_batch, x3_batch in dataloader:
                y_pred = self.network(x1=x1_batch, x3=x3_batch)
                probas = torch.nn.functional.softmax(y_pred, dim=1).numpy()
                all_probas.append(probas)

        return np.concatenate(all_probas, axis=0)

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
        # Minimal params (default architecture, fast training) ──
        params1 = {
            "num_epochs": 1,
            "batch_size": 2,
            "callbacks": None,
            "verbose": False,
            "random_state": 0,
        }

        # Explicit architecture with non-default values + Adam
        params2 = {
            "kernel_sizes": (3, 3),
            "rnn_layer": 16,
            "lstm_layer": 4,
            "evg_pool_size": 2,
            "n_conv_layers": 2,
            "filter_sizes": (8, 4),
            "dense_size": 32,
            "hidden_activation": "relu",
            "activation_attention": "sigmoid",
            "dropout": 0.5,
            "init_weights": "xavier_uniform",
            "num_epochs": 1,
            "batch_size": 2,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "callbacks": None,
            "optimizer_kwargs": None,
            "criterion_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }

        # RMSprop + tanh activation + kaiming init
        params3 = {
            "kernel_sizes": (1, 1),
            "rnn_layer": 32,
            "lstm_layer": 8,
            "evg_pool_size": 1,
            "n_conv_layers": 2,
            "filter_sizes": (16, 8),
            "dense_size": 64,
            "hidden_activation": "tanh",
            "activation_attention": "sigmoid",
            "dropout": 0.3,
            "init_weights": "kaiming_normal",
            "num_epochs": 1,
            "batch_size": 2,
            "optimizer": "RMSprop",
            "criterion": "CrossEntropyLoss",
            "callbacks": None,
            "optimizer_kwargs": None,
            "criterion_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.0005,
            "verbose": False,
            "random_state": 0,
        }

        # Per-layer dropout tuple + learning rate scheduler
        params4 = {
            "kernel_sizes": (1, 1),
            "rnn_layer": 64,
            "lstm_layer": 8,
            "evg_pool_size": 1,
            "n_conv_layers": 2,
            "filter_sizes": (16, 8),
            "dense_size": 64,
            "hidden_activation": "relu",
            "activation_attention": "sigmoid",
            "dropout": (0.8, 0.8, 0.7, 0.8, 0.6, 0.5, 0.8),  # per-layer
            "init_weights": "xavier_uniform",
            "num_epochs": 1,
            "batch_size": 2,
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "callbacks": "ReduceLROnPlateau",
            "optimizer_kwargs": {"weight_decay": 1e-4},
            "criterion_kwargs": None,
            "callback_kwargs": {"factor": 0.5, "patience": 10},
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }

        # No weight init + SGD with momentum + zero dropout
        params5 = {
            "kernel_sizes": (1, 1),
            "rnn_layer": 64,
            "lstm_layer": 8,
            "evg_pool_size": 1,
            "n_conv_layers": 2,
            "filter_sizes": (16, 8),
            "dense_size": 64,
            "hidden_activation": "relu",
            "activation_attention": "sigmoid",
            "dropout": 0.0,  # no dropout
            "init_weights": None,  # use PyTorch defaults
            "num_epochs": 1,
            "batch_size": 2,
            "optimizer": "SGD",
            "criterion": "CrossEntropyLoss",
            "callbacks": None,
            "optimizer_kwargs": {"momentum": 0.9},
            "criterion_kwargs": None,
            "callback_kwargs": None,
            "lr": 0.01,
            "verbose": False,
            "random_state": 0,
        }

        return [params1, params2, params3, params4, params5]
