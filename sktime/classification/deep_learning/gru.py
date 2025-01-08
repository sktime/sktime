"""Gated Recurrent Unit (GRU) for time series classification."""

import numpy as np

from sktime.classification.deep_learning._pytorch import BaseDeepClassifierPytorch
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    from sktime.networks.gru import GRU, GRUFCNN


class GRUClassifier(BaseDeepClassifierPytorch):
    """Gated Recurrent Unit (GRU) for time series classification.

    This classifier has been wrapped around implementations from [1]_, [2]_ and [3]_.

    Parameters
    ----------
    hidden_dim : int
        Number of features in the hidden state.
    n_layers : int
        Number of recurrent layers.
    batch_first : bool
        If True, then the input and output tensors are provided
        as (batch, seq, feature), default is False.
    bias : bool
        If False, then the layer does not use bias weights, default is True.
    init_weights : bool
        If True, then the weights are initialized, default is True.
    dropout : float
        Dropout rate to apply. default is 0.0
    fc_dropout : float
        Dropout rate to apply to the fully connected layer. default is 0.0
    bidirectional : bool
        If True, then the GRU is bidirectional, default is False.
    num_epochs : int, optional (default=10)
        The number of epochs to train the model.
    optimizer : str, optional (default=None)
        The optimizer to use. If None, Adam will be used.
    activation : str, optional (default="relu")
        The activation function to use. Options: ["relu", "softmax"].
    batch_size : int, optional (default=8)
        The size of each mini-batch during training.
    criterion : callable, optional (default=None)
        The loss function to use. If None, CrossEntropyLoss will be used.
    criterion_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    optimizer_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
    lr : float, optional (default=0.001)
        The learning rate to use for the optimizer.
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    random_state : int, optional (default=None)
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

    """

    _tags = {
        "authors": ["fnhirwa"],
        "maintainers": ["fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
    }

    def __init__(
        self: "GRUClassifier",
        # model specific
        hidden_dim: int = 256,
        n_layers: int = 4,
        batch_first: bool = False,
        bias: bool = True,
        init_weights: bool = True,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        bidirectional: bool = False,
        # base classifier specific
        num_epochs: int = 10,
        batch_size: int = 8,
        optimizer: str = None,
        criterion: str = None,
        criterion_kwargs: dict = None,
        optimizer_kwargs: dict = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int = None,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.bias = bias
        self.init_weights = init_weights
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.bidirectional = bidirectional
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # infer from the data
        self.input_size = None
        self.numclasses = None

        super().__init__(
            num_epochs=num_epochs,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=batch_size,
            criterion_kwargs=criterion_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            verbose=verbose,
            random_state=random_state,
        )

        self.criterions = {}

    def _build_network(self, X, y):
        # n_instances, n_dims, n_timesteps = X.shape
        self.numclasses = len(np.unique(y))
        _, self.input_size, _ = X.shape
        return GRU(
            input_size=self.input_size,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            batch_first=self.batch_first,
            bias=self.bias,
            num_classes=self.numclasses,
            init_weights=self.init_weights,
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
            bidirectional=self.bidirectional,
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
                "batch_first": False,
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
                "batch_first": False,
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
        ]
        return params


class GRUFCNNClassifier(BaseDeepClassifierPytorch):
    """GRU-FCN for time series classification.

    The network used in this classifier is originally defined in [1]_.
    The current implementation uses PyTorch and references
    the TensorFlow implementations in [2]_ and [3]_.

    Parameters
    ----------
    hidden_dim : int
        Number of features in the hidden state.
    gru_layers : int
        Number of recurrent layers.
    batch_first : bool
        If True, then the input and output tensors are provided
        as (batch, seq, feature), default is False.
    bias : bool
        If False, then the layer does not use bias weights, default is True.
    init_weights : bool
        If True, then the weights are initialized, default is True.
    dropout : float
        Dropout rate to apply inside gru cell. default is 0.0
    gru_dropout : float
        Dropout rate to apply to the gru output layer. default is 0.0
    bidirectional : bool
        If True, then the GRU is bidirectional, default is False.
    conv_layers : list
        List of integers specifying the number of filters in each convolutional layer.
        default is [128, 256, 128].
    kernel_sizes : list
        List of integers specifying the kernel size in each convolutional layer.
        default is [7, 5, 3].
    num_epochs : int, optional (default=10)
        The number of epochs to train the model.
    optimizer : str, optional (default=None)
        The optimizer to use. If None, Adam will be used.
    activation : str, optional (default="relu")
        The activation function to use. Options: ["relu", "softmax"].
    batch_size : int, optional (default=8)
        The size of each mini-batch during training.
    criterion : callable, optional (default=None)
        The loss function to use. If None, CrossEntropyLoss will be used.
    criterion_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    optimizer_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
    lr : float, optional (default=0.001)
        The learning rate to use for the optimizer.
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    random_state : int, optional (default=None)
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Elsayed, et al. "Deep Gated Recurrent and Convolutional Network Hybrid Model
        for Univariate Time Series Classification."
        arXiv preprint arXiv:1812.07683 (2018).
    .. [2] https://github.com/NellyElsayed/GRU-FCN-model-for-univariate-time-series-classification
    .. [3] https://github.com/titu1994/LSTM-FCN

    """

    _tags = {
        "authors": ["fnhirwa"],
        "maintainers": ["fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
    }

    def __init__(
        self: "GRUFCNNClassifier",
        # model specific
        hidden_dim: int,
        gru_layers: int,
        batch_first: bool = False,
        bias: bool = True,
        init_weights: bool = True,
        dropout: float = 0.0,
        gru_dropout: float = 0.0,
        bidirectional: bool = False,
        conv_layers: list = [128, 256, 128],
        kernel_sizes: list = [7, 5, 3],
        # base classifier specific
        num_epochs: int = 10,
        batch_size: int = 8,
        optimizer: str = "Adam",
        criterion: str = None,
        criterion_kwargs: dict = None,
        optimizer_kwargs: dict = None,
        lr: float = 0.01,
        verbose: bool = False,
        random_state: int = None,
    ):
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.batch_first = batch_first
        self.bias = bias
        self.init_weights = init_weights
        self.dropout = dropout
        self.gru_dropout = gru_dropout
        self.bidirectional = bidirectional
        self.conv_layers = conv_layers
        self.kernel_sizes = kernel_sizes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = {"betas": (0.9, 0.999)} if optimizer == "Adam" else {}
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

        # infer from the data
        self.input_size = None
        self.numclasses = None

        super().__init__(
            num_epochs=num_epochs,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=batch_size,
            criterion_kwargs=criterion_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
            verbose=verbose,
            random_state=random_state,
        )

        self.criterions = {}

    def _build_network(self, X, y):
        # n_instances, n_dims, n_timesteps = X.shape
        self.numclasses = len(np.unique(y))
        _, self.input_size, _ = X.shape
        return GRUFCNN(
            input_size=self.input_size,
            hidden_dim=self.hidden_dim,
            gru_layers=self.gru_layers,
            batch_first=self.batch_first,
            bias=self.bias,
            num_classes=self.numclasses,
            init_weights=self.init_weights,
            dropout=self.dropout,
            gru_dropout=self.gru_dropout,
            bidirectional=self.bidirectional,
            conv_layers=self.conv_layers,
            kernel_sizes=self.kernel_sizes,
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
                "gru_layers": 2,
                "batch_first": False,
                "bias": True,
                "init_weights": True,
                "dropout": 0.1,
                "gru_dropout": 0.1,
                "bidirectional": False,
                "conv_layers": [128, 256, 128],
                "kernel_sizes": [7, 5, 3],
                "num_epochs": 2,
                "optimizer": "Adam",
                "lr": 0.01,
                "verbose": False,
                "random_state": 0,
            },
            {
                "hidden_dim": 64,
                "gru_layers": 3,
                "batch_first": False,
                "bias": True,
                "init_weights": False,
                "dropout": 0.1,
                "gru_dropout": 0.0,
                "bidirectional": True,
                "conv_layers": [128, 256, 128],
                "kernel_sizes": [7, 5, 3],
                "num_epochs": 2,
                "optimizer": "Adam",
                "lr": 0.01,
                "verbose": False,
                "random_state": 0,
            },
        ]
        return params
