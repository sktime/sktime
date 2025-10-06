"""Time Recurrent Neural Network (RNN) for classification in PyTorch."""

__authors__ = ["RecreationalMath"]
__all__ = ["SimpleRNNClassifierTorch"]

import numpy as np

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.networks.rnn import RNNNetworkTorch
from sktime.utils.dependencies._safe_import import _safe_import

nnCrossEntropyLoss = _safe_import("torch.nn.CrossEntropyLoss")


class SimpleRNNClassifierTorch(BaseDeepClassifierPytorch):
    """Simple recurrent neural network in PyTorch for time series classification.

    Parameters
    ----------
    hidden_dim : int, default = 6
        Number of features in the hidden state.
    n_layers : int
        Number of recurrent layers.
        E.g., setting n_layers=2 would mean stacking two RNNs together to form
        a stacked RNN, with the second RNN taking in outputs of the first RNN
        and computing the final results.
    activation : str/callable
        The activation function to use. Can be either 'tanh' or 'relu'.
        Default is 'relu'.
    batch_first : bool, default = False
        If True, then the input and output tensors are provided
        as (batch, seq, feature).
    bias : bool, default = True
        If False, then the layer does not use bias weights.
    init_weights : bool, default = True
        If True, then the weights are initialized.
    dropout : float, default = 0.0
        If non-zero, introduces a Dropout layer on the outputs of each RNN layer
        except the last layer, with dropout probability equal to dropout.
    fc_dropout : float, default = 0.0
        If non-zero, introduces a Dropout layer on the outputs of the
        fully connected layer, with dropout probability equal to fc_dropout.
    bidirectional : bool, default = False
        If True, then the RNN is bidirectional.
    num_epochs : int, optional (default=100)
        The number of epochs to train the model.
    optimizer : case insensitive str or an instance of optimizers
        defined in torch.optim, optional (default = "RMSprop").
        The optimizer to use for training the model.
        List of available optimizers:
        https://pytorch.org/docs/stable/optim.html#algorithms
    batch_size : int, optional (default=1)
        The size of each mini-batch during training.
    criterion : case insensitive str, or an instance of a loss function
        defined in PyTorch, optional (default="CrossEntropyLoss").
        The loss function to be used in training the neural network.
        List of available loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions
    criterion_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the loss function.
    optimizer_kwargs : dict, optional (default=None)
        Additional keyword arguments to pass to the optimizer.
    lr : float, optional (default=0.001)
        The learning rate to use for the optimizer.
    verbose : bool, optional (default=False)
        Whether to print progress information during training.
    random_state : int, optional (default=0)
        Seed to ensure reproducibility.
    metrics : list of str, default = ["accuracy"]
        List of metrics to be used for evaluation.

    Examples
    --------
    >>> from sktime.classification.deep_learning.rnn import SimpleRNNClassifierTorch
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>> clf = SimpleRNNClassifierTorch(n_epochs=50,batch_size=2) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    SimpleRNNClassifierTorch(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["RecreationalMath"],
        "maintainers": ["RecreationalMath"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "capability:random_state": True,
    }

    def __init__(
        self: "SimpleRNNClassifierTorch",
        # model specific
        hidden_dim: int = 6,
        n_layers: int = 1,
        activation: str = "relu",
        batch_first: bool = False,
        bias: bool = True,
        init_weights: bool = True,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        bidirectional: bool = False,
        # base classifier specific
        num_epochs: int = 100,
        batch_size: int = 1,
        optimizer: str = "RMSprop",
        criterion: str = "CrossEntropyLoss",
        criterion_kwargs: dict = None,
        optimizer_kwargs: dict = None,
        lr: float = 0.001,
        verbose: bool = False,
        random_state: int = 0,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
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
        self.num_classes = None

        super().__init__(
            num_epochs=self.num_epochs,
            optimizer=self.optimizer,
            criterion=self.criterion,
            batch_size=self.batch_size,
            criterion_kwargs=self.criterion_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _build_network(self, X, y):
        """Build the RNN network.

        Parameters
        ----------
        X : numpy.ndarray
            Input data containing the time series data.
        y : numpy.ndarray
            Target labels for the classification task.

        Returns
        -------
        model : RNNNetworkTorch instance
            The constructed RNN network.
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. Please ensure your input data is "
                "properly formatted."
            )
        # n_instances, n_dims, n_timesteps = X.shape
        self.num_classes = len(np.unique(y))
        _, self.input_size, _ = X.shape
        return RNNNetworkTorch(
            input_size=self.input_size,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            activation=self.activation,
            bias=self.bias,
            batch_first=self.batch_first,
            num_classes=self.num_classes,
            init_weights=self.init_weights,
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
            bidirectional=self.bidirectional,
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
        params1 = {}
        params2 = {
            "hidden_dim": 5,
            "n_layers": 1,
            "activation": "relu",
            "batch_first": False,
            "bias": False,
            "init_weights": True,
            "dropout": 0.0,
            "fc_dropout": 0.0,
            "bidirectional": False,
            "num_epochs": 50,
            "batch_size": 2,
            "optimizer": "RMSprop",
            "criterion": None,
            "criterion_kwargs": None,
            "optimizer_kwargs": None,
            "lr": 0.001,
            "verbose": False,
            "random_state": 0,
        }

        return [params1, params2]
