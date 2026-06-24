"""ConvTran Neural Network for classification in PyTorch."""

__authors__ = ["srupat"]
__all__ = ["ConvTranClassifierTorch"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.convtran import ConvTranNetworkTorch


class ConvTranClassifierTorch(BaseDeepClassifierPytorch):
    """ConvTran classifier in PyTorch.

    Parameters
    ----------
    net_type : str, default="C-T"
        Network type to use. Should be one of "T" (Transformer),
        "C-T" (ConvTran) or "C-CT" (Causal ConvTran).
    activation : str or None, default=None
        Activation function to use in the output layer.
    activation_hidden : str, default="relu"
        Activation function to use in the hidden layers.
    emb_size : int, default=16
        Embedding dimension used in attention and feed-forward blocks.
    dim_ff : int, default=256
        Hidden dimension of the feed-forward block.
    num_heads : int, default=8
        Number of attention heads.
    dropout : float, default=0.01
        Dropout rate applied in attention and feed-forward blocks.
    use_abs_pos_encoding : bool, default=True
        Whether to apply absolute positional encoding.
    use_rel_pos_encoding : bool, default=True
        Whether to apply relative positional encoding.
    abs_pos_encoding_scheme : str or None, default="tAPE"
        Absolute positional encoding scheme. Supported values:
        "tAPE", "sin", "learn", or None.
    rel_pos_encoding_scheme : str or None, default="erpe"
        Relative positional encoding scheme. Supported values:
        "erpe", "vector", or None.
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
    >>> from sktime.classification.deep_learning.convtran import ConvTranClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = ConvTranClassifierTorch(num_epochs=20, batch_size=4)  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    ConvTranClassifierTorch(...)
    """

    _tags = {
        "authors": ["srupat"],
        "maintainers": ["srupat"],
        "python_version": ">=3.10, <3.15",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
        "capability:multivariate": True,
        # CI and test tags
        # ----------------
        "tests:vm": True,
    }

    def __init__(
        self: "ConvTranClassifierTorch",
        # model specific
        net_type: str = "C-T",
        activation: str | None | Callable = None,
        activation_hidden: str = "relu",
        emb_size: int = 16,
        dim_ff: int = 256,
        num_heads: int = 8,
        dropout: float = 0.01,
        use_abs_pos_encoding: bool = True,
        use_rel_pos_encoding: bool = True,
        abs_pos_encoding_scheme: str | None = "tAPE",
        rel_pos_encoding_scheme: str | None = "erpe",
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
        self.net_type = net_type
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.emb_size = emb_size
        self.dim_ff = dim_ff
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_abs_pos_encoding = use_abs_pos_encoding
        self.use_rel_pos_encoding = use_rel_pos_encoding
        self.abs_pos_encoding_scheme = abs_pos_encoding_scheme
        self.rel_pos_encoding_scheme = rel_pos_encoding_scheme
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
        """Build the ConvTran network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels corresponding to the input data.

        Returns
        -------
        model : ConvTranNetworkTorch
            An instance of the ConvTranNetworkTorch class initialized with the
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
        return ConvTranNetworkTorch(
            input_size=self.input_size,
            num_classes=self.num_classes,
            net_type=self.net_type,
            activation=self._validated_activation,
            activation_hidden=self.activation_hidden,
            emb_size=self.emb_size,
            dim_ff=self.dim_ff,
            num_heads=self.num_heads,
            dropout=self.dropout,
            use_abs_pos_encoding=self.use_abs_pos_encoding,
            use_rel_pos_encoding=self.use_rel_pos_encoding,
            abs_pos_encoding_scheme=self.abs_pos_encoding_scheme,
            rel_pos_encoding_scheme=self.rel_pos_encoding_scheme,
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
            "num_epochs": 2,
            "batch_size": 4,
            "net_type": "C-T",
            "emb_size": 8,
            "dim_ff": 16,
            "num_heads": 2,
            "abs_pos_encoding_scheme": "tAPE",
            "rel_pos_encoding_scheme": "erpe",
        }
        params3 = {
            "num_epochs": 2,
            "batch_size": 4,
            "net_type": "T",
            "emb_size": 8,
            "dim_ff": 16,
            "num_heads": 2,
            "abs_pos_encoding_scheme": "sin",
            "rel_pos_encoding_scheme": "erpe",
        }
        params4 = {
            "num_epochs": 2,
            "batch_size": 4,
            "net_type": "C-CT",
            "emb_size": 8,
            "dim_ff": 16,
            "num_heads": 2,
            "abs_pos_encoding_scheme": "learn",
            "use_rel_pos_encoding": False,
        }
        return [params1, params2, params3, params4]
