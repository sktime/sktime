"""ConvTimeNet (PyTorch) classifier for time series classification."""

__authors__ = ["Tanuj-Taneja1"]
__all__ = ["ConvTimeNetClassifier"]

from collections.abc import Callable

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.convtimenet import ConvTimeNetNetworkTorch


class ConvTimeNetClassifier(BaseDeepClassifierPytorch):
    """ConvTimeNet for time series classification.

    ConvTimeNet is a hierarchical pure convolutional model designed.
    Unlike prevalent methods centered around self-attention mechanisms,
    ConvTimeNet introduces two key innovations:

    1. A deformable patch layer that adaptively perceives local patterns of
       temporally dependent basic units in a data-driven manner.
    2. Hierarchical pure convolutional blocks that capture dependency relationships
        among the representations of basic units at different scales.

    The model employs a large kernel mechanism allowing convolutional blocks
    to be deeply stacked, achieving a larger receptive field. This architecture
    effectively models both local patterns and their multi-scale dependencies
    within a single model, addressing common challenges in time series analysis
    such as adaptive perception of local patterns and multi-scale dependency capture.

    This classifier has been wrapped around implementations from [1]_ and [2]_.

    Parameters
    ----------
    d_model : int, optional (default=64)
        Hidden dimension size for model processing.
    patch_size : int, optional (default=4)
        Size of patches for sequence splitting.
    patch_stride : int, optional (default=2)
        Stride length for patch creation.
    dropout : float, optional (default=0)
        Dropout rate to apply to layers.
    d_ff : int, optional (default=128)
        Dimension of feedforward network.
    dw_ks : int or list, optional (default=3)
        Depthwise convolution kernel size(s). Can be a single int or list of ints.
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

    activation_hidden : str, Callable, or None, default="gelu"
        Activation applied to the hidden layers, i.e., the deformable patch
        layer and the convolutional blocks.

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

    device : str, optional (default="cpu")
        Device to use for computation ("cpu" or "cuda").
    num_epochs : int, optional (default=16)
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
    random_state : int, optional (default=None)
        Seed to ensure reproducibility.

    Examples
    --------
    >>> from sktime.classification.deep_learning import ConvTimeNetClassifier
    >>> import numpy as np
    >>> # Create a sample multivariate time series dataset
    >>> # 48 samples, 3 variables, length 128
    >>> X = np.random.randn(16 * 3, 3, 128).astype("float32")
    >>> y = np.array([0, 1, 2] * 16)  # 3 classes
    >>> # Create and fit the classifier
    >>> clf = ConvTimeNetClassifier(
    ...     patch_size=4,
    ...     patch_stride=2,
    ...     d_model=64,
    ...     d_ff=128,
    ...     dw_ks=[5, 7, 9],
    ...     batch_size=8,
    ...     device="cpu",
    ...     random_state=10
    ... ) # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    ConvTimeNetClassifier(...)
    >>> # Make predictions
    >>> y_pred = clf.predict(X)  # doctest: +SKIP
    >>> y_proba = clf.predict_proba(X)  # doctest: +SKIP

    References
    ----------
    .. [1] Cheng, M., Yang, J., Pan, T., Liu, Q., & Li, Z. (2024). ConvTimeNet: A deep
        hierarchical fully convolutional model for multivariate time series analysis.
        arXiv preprint arXiv:2403.01493. https://arxiv.org/abs/2403.01493
    .. [2] https://github.com/Mingyue-Cheng/ConvTimeNet
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Mingyue-Cheng", "0russewt0", "pty12345", "Tanuj-Taneja1"],
        "maintainers": ["Tanuj-Taneja1"],
        "python_dependencies": ["torch"],
        # estimator type
        # --------------
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # CI and testing
        # --------------
        "tests:vm": True,
        "tests:libs": [
            "sktime.networks.convtimenet._convtimenet_torch",
            "sktime.networks.convtimenet._dlutils",
            "sktime.networks.convtimenet._convtimenet_backbone",
        ],
    }

    def __init__(
        self: "ConvTimeNetClassifier",
        # model specific
        d_model: int = 64,
        patch_size: int = 4,
        patch_stride: int = 2,
        dropout: float = 0,
        d_ff: int = 128,
        dw_ks: int | list[int] = 3,
        activation: str | None | Callable = None,
        activation_hidden: str | None | Callable = "gelu",
        device: str = "cpu",
        # base classifier specific
        num_epochs: int = 16,
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
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.dropout = dropout
        self.d_ff = d_ff
        self.dw_ks = dw_ks
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.device = device
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
        # Ensure dw_ks is a list
        if isinstance(self.dw_ks, int):
            self._dw_ks = [self.dw_ks]
        else:
            self._dw_ks = list(self.dw_ks)

        # enc_in, seq_len and num_classes are inferred from the data
        # and will be set in _build_network
        self.enc_in = None
        self.seq_len = None
        self.num_classes = None

        super().__post_init__()

    def _build_network(self, X, y):
        """Build the ConvTimeNet network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels corresponding to the input data.

        Returns
        -------
        model : ConvTimeNetNetworkTorch
            An instance of the ConvTimeNetNetworkTorch class initialized with the
            appropriate parameters.
        """
        self.num_classes = len(np.unique(y))
        self.enc_in = X.shape[1]
        self.seq_len = X.shape[2]

        model = ConvTimeNetNetworkTorch(
            enc_in=self.enc_in,
            d_model=self.d_model,
            seq_len=self.seq_len,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            n_classes=self.num_classes,
            dropout=self.dropout,
            d_ff=self.d_ff,
            dw_ks=self._dw_ks,
            activation=self._callable_activations["activation"],
            activation_hidden=self._callable_activations["activation_hidden"],
            device=self.device,
        )
        return model.to(self.device)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        params1 = {
            "d_model": 16,
            "patch_size": 2,
            "patch_stride": 1,
            "dw_ks": [3],
            "d_ff": 16,
            "batch_size": 2,
            "optimizer": "Adam",
            "lr": 1e-3,
            "device": "cpu",
            "verbose": False,
            "dropout": 0.0,
            "num_epochs": 1,
            "random_state": 0,
        }

        params2 = {
            "d_model": 32,
            "patch_size": 4,
            "patch_stride": 2,
            "dw_ks": [5, 7],
            "d_ff": 64,
            "batch_size": 4,
            "optimizer": "Adam",
            "lr": 5e-4,
            "device": "cpu",
            "verbose": False,
            "dropout": 0.1,
            "num_epochs": 2,
            "random_state": 42,
            "activation_hidden": "relu",
            "callbacks": "ReduceLROnPlateau",
        }

        params3 = {
            "d_model": 64,
            "patch_size": 5,
            "patch_stride": 1,
            "dw_ks": [7, 13, 19],  # very large depthwise kernels
            "d_ff": 128,
            "batch_size": 8,
            "optimizer": "SGD",  # different optimizer
            "criterion": "NLLLoss",
            "activation": "logsoftmax",
            "lr": 1e-2,
            "device": "cpu",
            "verbose": False,
            "dropout": 0.2,
            "num_epochs": 2,
            "random_state": 123,
        }

        return [params1, params2, params3]
