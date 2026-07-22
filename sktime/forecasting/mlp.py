"""MLP (Multi-Layer Perceptron) Forecaster for Time Series Forecasting.

Implementation of MLP benchmark from M4 forecasting competition.
"""

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.networks.mlp import MLPForecastNetwork
from sktime.utils.dependencies import _check_soft_dependencies


class MLPForecaster(BaseDeepNetworkPyTorch):
    """Multi-Layer Perceptron (MLP) Forecaster.

    Simple feedforward neural network for time series forecasting,
    as used in the M4 forecasting competition benchmarks [1]_.

    The model takes a fixed-length sequence of past values and outputs
    predictions for the specified forecast horizon.

    Parameters
    ----------
    seq_len : int
        Length of input sequence (lookback window).
    hidden_dim : int, default=100
        Number of units in each hidden layer.
    n_layers : int, default=3
        Number of hidden layers.
    dropout : float, default=0.0
        Dropout rate between hidden layers.
    batch_size : int, default=32
        Number of training examples per batch.
    num_epochs : int, default=100
        Number of training epochs.
    criterion : str, default=None
        Loss function. One of "MSE", "L1", "SmoothL1", "Huber".
        If None, uses MSE.
    criterion_kwargs : dict, default=None
        Keyword arguments for the loss function.
    optimizer : str, default="Adam"
        Optimizer for training. One of "Adadelta", "Adagrad", "Adam", "AdamW", "SGD".
    optimizer_kwargs : dict, default=None
        Keyword arguments for the optimizer.
    lr : float, default=1e-3
        Learning rate for training.
    custom_dataset_train : Dataset, default=None
        Custom PyTorch Dataset for training.
    custom_dataset_pred : Dataset, default=None
        Custom PyTorch Dataset for prediction.

    References
    ----------
    .. [1] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. 2018.
    The M4 Competition: Results, findings, conclusion and way forward.
    International Journal of Forecasting.
    https://github.com/Mcompetitions/M4-methods

    Examples
    --------
    >>> from sktime.forecasting.mlp import MLPForecaster  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = MLPForecaster(seq_len=12, num_epochs=10)  # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        "authors": ["fkiraly"],
        "maintainers": ["fkiraly"],
        "python_dependencies": ["torch"],
        # estimator type
        "y_inner_mtype": "pd.DataFrame",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "scitype:y": "both",
        "capability:exogenous": False,
    }

    def __init__(
        self,
        seq_len=10,
        hidden_dim=100,
        n_layers=3,
        dropout=0.0,
        batch_size=32,
        num_epochs=100,
        criterion=None,
        criterion_kwargs=None,
        optimizer="Adam",
        optimizer_kwargs=None,
        lr=1e-3,
        custom_dataset_train=None,
        custom_dataset_pred=None,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.criterion = criterion
        self.custom_dataset_train = custom_dataset_train
        self.custom_dataset_pred = custom_dataset_pred

        super().__init__(
            num_epochs=num_epochs,
            batch_size=batch_size,
            criterion_kwargs=criterion_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=lr,
        )

        if _check_soft_dependencies("torch", severity="none"):
            import torch

            self.criterions = {
                "MSE": torch.nn.MSELoss,
                "L1": torch.nn.L1Loss,
                "SmoothL1": torch.nn.SmoothL1Loss,
                "Huber": torch.nn.HuberLoss,
            }

            self.optimizers = {
                "Adadelta": torch.optim.Adadelta,
                "Adagrad": torch.optim.Adagrad,
                "Adam": torch.optim.Adam,
                "AdamW": torch.optim.AdamW,
                "SGD": torch.optim.SGD,
            }

    def _build_network(self, fh):
        """Build the MLP network.

        Parameters
        ----------
        fh : int
            Forecast horizon length.

        Returns
        -------
        nn.Module
            Initialized MLP network.
        """
        return MLPForecastNetwork(
            seq_len=self.seq_len,
            pred_len=fh,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )._build()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {
            "seq_len": 3,
            "hidden_dim": 8,
            "n_layers": 1,
            "num_epochs": 2,
            "batch_size": 4,
        }
        params2 = {
            "seq_len": 4,
            "hidden_dim": 16,
            "n_layers": 2,
            "dropout": 0.1,
            "num_epochs": 3,
            "batch_size": 2,
            "optimizer": "Adam",
            "lr": 1e-3,
        }
        return [params1, params2]
