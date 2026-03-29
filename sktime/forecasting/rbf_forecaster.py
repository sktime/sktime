"""RBF Neural Networks for Time Series Forecasting."""

__author__ = ["phoeenniixx"]

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.networks.rbf import RBFNetwork
from sktime.utils.dependencies._dependencies import _check_soft_dependencies
from sktime.utils.warnings import warn


class RBFForecaster(BaseDeepNetworkPyTorch):
    r"""Forecasting model using RBF transformations and 'NN' layers for time series.

    This forecaster uses an RBF layer to transform input time series data into a
    higher-dimensional space, which is then used by neural network layers for
    forecasting.

     Model Architecture:
    -------------------

    1. Input layer:

       - Takes in the input time series data

    2. RBF Layer:

       - Applies Radial Basis Function (RBF) transformation to the input features
       - Uses centers and gamma parameter to control the spread of the RBFs
       - Expands the input feature space into a higher-dimensional representation

    3. Sequential layers:

       - One or more fully connected linear layer specified in `hidden_layers`.
       - Non-linear activation (e.g., ReLU, GELU) applied after each layer.
       - Dropout with specified rate applied after each layer.
       - The final linear layer produces the output prediction

    Parameters
    ----------
    window_length : int, optional (default=10)
        Length of the input sequence for each sliding window.
    hidden_size : int, optional (default=32)
        Number of units in the RBF layer.
    batch_size : int, optional (default=32)
        Size of mini-batches for training.
    centers : array-like, optional (default=None)
        Center points for RBF transformations.
    gamma : float, optional (default=1.0)
        Scaling factor controlling the spread of the RBF.
    rbf_type : str, optional (default="gaussian")
        The type of RBF kernel to apply.

        - "gaussian": :math:`\exp(-\gamma (t - c)^2)`
        - "multiquadric": :math:`\sqrt{1 + \gamma (t - c)^2}`
        - "inverse_multiquadric": :math:`\frac{1}{\sqrt{1 + \gamma (t - c)^2}}`
    hidden_layers : list of int, optional (default=[64, 32])
        Sizes of linear layers following the RBF layer.
    optimizer : {"adam", "sgd", "rmsprop"}, optional (default="adam")
        Type of optimizer to use.
    lr : float, optional (default=0.01)
        Learning rate for optimizer.
    epochs : int, optional (default=100)
        Number of training epochs.
    stride : int, optional (default=1)
        Step size between windows.
    criterion : str, optional (default="mse")
        Loss function to use during training.
    device : str, optional (default="cpu")
        Device to use for training and computation. Options are "cpu" or "cuda"
        for GPU computation if available.
    mode : {"ar", "direct"}, optional (default="ar")
        Forecasting mode:

        - "ar": Autoregressive mode for one-step-ahead predictions.
        - "direct": Direct mode for multi-step-ahead predictions.
    pred_len : int, optional (default=None)
        Prediction length, i.e., the number of future time steps to forecast.
        Defines the network output dimension in direct mode.
        In AR mode this is ignored (output is always 1).
        Required for pretraining in direct mode if fh is not passed to pretrain().
    activation : str, optional (default="relu")
        Activation function to apply after each linear layer. Supported values are:
        "relu", "leaky_relu", "elu", "selu", "tanh", "sigmoid", "gelu".
    dropout_rate : float, optional (default=0.1)
        Dropout rate applied after each hidden layer. A value of 0 disables dropout.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["phoeenniixx"],
        "maintainers": ["phoeenniixx"],
        "python_dependencies": "torch",
        # estimator type
        # --------------
        "capability:multivariate": False,
        "y_inner_mtype": "pd.Series",
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "capability:pretrain": True,
    }

    def __init__(
        self,
        window_length=10,
        hidden_size=32,
        batch_size=32,
        centers=None,
        gamma=1.0,
        rbf_type="gaussian",
        hidden_layers=[64, 32],
        optimizer="adam",
        lr=0.01,
        epochs=100,
        stride=1,
        criterion="mse",
        device="cpu",
        mode="ar",
        pred_len=None,
        activation="relu",
        dropout_rate=0.1,
    ):
        super().__init__()

        self.window_length = window_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.centers = centers
        self.gamma = gamma
        self.rbf_type = rbf_type
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.stride = stride
        self.criterion = criterion
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.mode = mode
        self.pred_len = pred_len
        self._fh_length = None

        if self.mode not in ["ar", "direct"]:
            raise ValueError("mode must be either 'ar' or 'direct'")

        self.device = device
        self._device = self._get_device(device)
        self.network = None
        self.scaler = StandardScaler()

    def _get_device(self, device):
        """Convert device parameter to torch.device object.

        Parameters
        ----------
        device : str or torch.device
            Device specification

        Returns
        -------
        torch.device
            Initialized device object
        """
        if _check_soft_dependencies("torch", severity="none"):
            import torch
        if isinstance(device, str):
            if device not in ["cpu", "cuda", "mps"]:
                raise ValueError("device must be 'cpu', 'cuda', or 'mps'")
            if device == "cuda" and not torch.cuda.is_available():
                warn("CUDA is not available, using CPU instead", UserWarning)
                device = "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                warn("MPS is not available, using CPU instead", UserWarning)
                device = "cpu"
        return torch.device(device)

    def _build_network(self, fh):
        """Build the RBF network architecture.

        Parameters
        ----------
        fh : int
            Prediction length (output dimension for direct mode, ignored for AR).
        """
        output_size = fh if self.mode == "direct" else 1

        return RBFNetwork(
            input_size=self.window_length,
            hidden_size=self.hidden_size,
            output_size=output_size,
            centers=self.centers,
            gamma=self.gamma,
            rbf_type=self.rbf_type,
            hidden_layers=self.hidden_layers,
            mode=self.mode,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
        ).to(self._device)

    def _fit(self, y, fh, X=None):
        """Fit the RBF-based model to the provided time series data.

        Parameters
        ----------
        y : pd.Series
            The time series data to be fitted.
        fh : ForecastingHorizon
            Forecasting horizon.
        X : optional
            Additional exogenous data (not used).
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._y = y.copy()
        self._y_len = len(y)

        if isinstance(y, pd.Series):
            y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1))
        else:
            y_scaled = self.scaler.fit_transform(y.values)

        if self.mode == "direct":
            self._fh_length = len(fh) if fh is not None else 1
        else:
            self._fh_length = 1

        if hasattr(self, "network") and self.network is not None:
            # Pretrained network exists: validate fh for direct mode only
            # AR mode has no fh constraint (iterates step-by-step)
            if self.mode == "direct" and fh is not None:
                fh_rel = fh.to_relative(self.cutoff)
                max_fh = max(list(fh_rel))
                if max_fh > self.network.pred_len:
                    raise ValueError(
                        f"max(fh)={max_fh} exceeds the network's output "
                        f"dimension (pred_len={self.network.pred_len}). "
                        f"The network architecture was fixed during "
                        f"pretraining. Either use a smaller fh "
                        f"(<= {self.network.pred_len}) or create a new "
                        f"forecaster with a larger pred_len."
                    )
                self._fh_length = self.network.pred_len
        else:
            self.network = self._build_network(self._fh_length)

        X_train, y_train = self._create_windows(y_scaled)
        X_tensor = torch.FloatTensor(X_train).to(self._device)
        y_tensor = torch.FloatTensor(y_train).to(self._device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        self.network.train()
        for epoch in range(self.epochs):
            self._run_epoch(epoch, dataloader)

        return self

    def _predict(self, fh=None, X=None):
        """Generate predictions for the specified forecasting horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon for the predictions.
        X : optional
            Additional exogenous data (not used).
        """
        import torch

        fh_abs = fh.to_absolute(self._y.index[-1]).to_numpy()
        n_steps = len(fh_abs)
        predictions = np.zeros(n_steps)

        last_window = self._y[-self.window_length :].copy()
        current_window = self.scaler.transform(last_window.values.reshape(-1, 1))

        self.network.eval()
        with torch.no_grad():
            if self.mode == "direct":
                # Direct prediction: predict all steps at once
                X_predict = torch.FloatTensor(current_window.reshape(1, -1)).to(
                    self._device
                )
                pred_scaled = self.network(X_predict).cpu().numpy()
                predictions = self.scaler.inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).ravel()
            else:
                # AR mode: iterate step by step
                for i in range(n_steps):
                    X_predict = torch.FloatTensor(current_window.reshape(1, -1)).to(
                        self._device
                    )
                    pred_scaled = self.network(X_predict).cpu().numpy()
                    pred = self.scaler.inverse_transform(pred_scaled.reshape(1, -1))
                    predictions[i] = pred.ravel()[0]

                    current_window = np.roll(current_window, -1, axis=0)
                    current_window[-1] = pred_scaled.ravel()[0]

        return pd.Series(predictions, index=fh_abs, name=self._y.name)

    def _pretrain(self, y, X=None, fh=None):
        """Pretrain the RBF network on panel data.

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex
            Panel data to pretrain on.
        X : pd.DataFrame, optional
            Exogenous data (not used).
        fh : ForecastingHorizon, optional
            Forecasting horizon. Uses pred_len from constructor if not provided.
        """
        from sktime.forecasting.base.adapters._pytorch import _get_series_from_panel

        pred_len = self._get_pretrain_pred_len(fh)
        all_series = _get_series_from_panel(y)

        # Use first column of first series as reference (univariate forecaster)
        ref_series = all_series[0]
        if isinstance(ref_series, pd.DataFrame) and ref_series.shape[1] > 1:
            ref_series = ref_series.iloc[:, 0]
        self._y = ref_series
        self._y_len = len(ref_series)
        self._fh_length = pred_len if self.mode == "direct" else 1

        self.network = self._build_network(pred_len)
        dataloader = self._build_panel_dataloader(y, all_series, pred_len)

        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        self.network.train()
        for epoch in range(self.epochs):
            self._run_epoch(epoch, dataloader)

        self._store_pretrain_metadata(y, pred_len)
        return self

    def _pretrain_update(self, y, X=None, fh=None):
        """Update pretrained RBF network with additional panel data.

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex
            Additional panel data to train on.
        X : pd.DataFrame, optional
            Exogenous data (not used).
        fh : ForecastingHorizon, optional
            Forecasting horizon.
        """
        from sktime.forecasting.base.adapters._pytorch import _get_series_from_panel

        if hasattr(self, "_pretrain_pred_len"):
            pred_len = self._pretrain_pred_len
        else:
            pred_len = self._get_pretrain_pred_len(fh)

        all_series = _get_series_from_panel(y)
        dataloader = self._build_panel_dataloader(y, all_series, pred_len)

        self.network.train()
        for epoch in range(self.epochs):
            self._run_epoch(epoch, dataloader)

        if hasattr(y, "index") and isinstance(y.index, pd.MultiIndex):
            n_new = len(y.index.get_level_values(0).unique())
            if hasattr(self, "n_pretrain_instances_"):
                self.n_pretrain_instances_ += n_new
            else:
                self.n_pretrain_instances_ = n_new

        return self

    def _build_panel_dataloader(self, y, all_series, pred_len):
        """Build PyTorch DataLoader for panel data pretraining.

        Parameters
        ----------
        y : pd.DataFrame with MultiIndex
            Panel data (not used directly).
        all_series : list of pd.DataFrame
            Pre-extracted individual time series.
        pred_len : int
            Prediction length for the dataset.
        """
        import torch
        from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

        fh_length = pred_len if self.mode == "direct" else 1
        self._fh_length = fh_length

        # Split multi-column DataFrames into separate univariate series
        univariate_series = []
        for series in all_series:
            if isinstance(series, pd.DataFrame) and series.shape[1] > 1:
                for col in series.columns:
                    univariate_series.append(series[[col]])
            else:
                univariate_series.append(series)

        datasets = []
        for series in univariate_series:
            scaler = StandardScaler()
            if isinstance(series, pd.Series):
                y_scaled = scaler.fit_transform(series.values.reshape(-1, 1))
            else:
                y_scaled = scaler.fit_transform(series.values)

            X_train, y_train = self._create_windows(y_scaled)
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.FloatTensor(y_train)
            datasets.append(TensorDataset(X_tensor, y_tensor))

        combined_dataset = ConcatDataset(datasets)
        return DataLoader(combined_dataset, self.batch_size, shuffle=True)

    def _get_pretrain_pred_len(self, fh):
        """Get prediction length for pretraining.

        AR mode always returns 1 since the network outputs a single step.
        Direct mode uses pred_len from constructor or fh.
        """
        if self.mode == "ar":
            return 1
        if hasattr(self, "pred_len") and self.pred_len is not None:
            return int(self.pred_len)
        elif fh is not None:
            if isinstance(fh, (int, float)):
                return int(fh)
            return int(list(fh)[-1])
        else:
            raise ValueError(
                "pred_len must be specified either in constructor or via fh "
                "parameter for pretraining in direct mode."
            )

    def _get_seq_len(self):
        """Return sequence length (window_length for RBF)."""
        return self.window_length

    def _create_windows(self, y):
        """Generate sliding windows from the time series data.

        This function generates sliding windows of input features and corresponding
        targets for time series data. It supports two modes:

          - "direct": Outputs windows with multiple target values for
                direct multi-step prediction.
          - "AR" (Auto-Regressive): Outputs windows with a single target
                value for one-step prediction.

        Parameters
        ----------
        y : np.ndarray
            The input time series array.

        Returns
        -------
        tuple of np.ndarray

            - windows_list: An array containing input windows.
            - targets_list: An array containing the corresponding target values.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = len(y)
        windows_list = []
        targets_list = []

        if self.mode == "direct":
            # For direct mode, create windows with multiple target values
            for i in range(
                0, n_samples - self.window_length - self._fh_length + 1, self.stride
            ):
                window = y[i : (i + self.window_length)]
                target = y[
                    (i + self.window_length) : (
                        i + self.window_length + self._fh_length
                    )
                ]

                if len(target) == self._fh_length:
                    windows_list.append(window.flatten())
                    targets_list.append(target.flatten())
        else:
            # For AR mode, create windows with single target value
            for i in range(0, n_samples - self.window_length, self.stride):
                window = y[i : (i + self.window_length)]
                target = y[i + self.window_length]

                windows_list.append(window.flatten())
                targets_list.append(target)
        fhlen = self._fh_length if self.mode == "direct" else 1
        if not windows_list:
            raise ValueError(
                f"Not enough samples to create windows. Need at least "
                f"{self.window_length + fhlen} "
                f"samples, but got {n_samples}"
            )

        return np.array(windows_list), np.array(targets_list)

    def _instantiate_optimizer(self):
        from torch.optim import SGD, Adam, RMSprop

        optimizers = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
        if self.optimizer.lower() not in optimizers:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        return optimizers[self.optimizer.lower()](self.network.parameters(), lr=self.lr)

    def _instantiate_criterion(self):
        import torch.nn as nn

        loss_fns = {
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "poisson": nn.PoissonNLLLoss,
            "bce": nn.BCELoss,
            "crossentropy": nn.CrossEntropyLoss,
        }
        if self.criterion.lower() not in loss_fns:
            raise ValueError(f"Unsupported Loss Function: {self.criterion}")
        return loss_fns[self.criterion.lower()]()

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Provide example parameters for unit testing or experimentation.

        Returns
        -------
        params : dict or list of dict
        """
        # window_length + pred_len must be < min_timepoints (10) in pretrain tests
        params_list = [
            {
                "window_length": 5,
                "hidden_size": 8,
                "batch_size": 16,
                "gamma": 1.0,
                "rbf_type": "gaussian",
                "hidden_layers": [16],
                "epochs": 3,
                "lr": 0.01,
                "stride": 1,
                "optimizer": "adam",
                "device": "cpu",
                "mode": "ar",
                "activation": "relu",
                "dropout_rate": 0.1,
            },
            {
                "window_length": 5,
                "hidden_size": 32,
                "batch_size": 32,
                "gamma": 0.5,
                "rbf_type": "gaussian",
                "hidden_layers": [64, 32],
                "epochs": 3,
                "lr": 0.005,
                "stride": 1,
                "optimizer": "adam",
                "criterion": "mse",
                "device": "cpu",
                "mode": "direct",
                "pred_len": 3,
                "activation": "gelu",
                "dropout_rate": 0.2,
            },
        ]

        return params_list
