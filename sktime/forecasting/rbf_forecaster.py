"""RBF Neural Networks for Time Series Forecasting."""

__author__ = ["phoeenniixx"]

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.utils.dependencies._dependencies import _check_soft_dependencies
from sktime.utils.warnings import warn

if _check_soft_dependencies("torch", severity="warning"):
    import torch
    import torch.nn as nn

else:

    class nn:
        """dummy class if torch is not available."""

        class Module:
            """dummy class if torch is not available."""

            def __init__(self, *args, **kwargs):
                raise ImportError("torch is not available. Please install torch first.")


class RBFLayer(nn.Module):
    """RBF layer to transform input data into a new feature space.

    This layer applies an RBF transformation to each input feature,
    expanding the feature space based on distances to predefined center points.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features (number of RBF centers)
    centers : torch.Tensor, optional (default=None)
        The centers :math:`c_k` for the RBF transformation.
        If None, centers are evenly spaced.
    gamma : float, optional (default=1.0)
        Parameter controlling the spread of the RBFs
    rbf_type :{"gaussian", "multiquadric", "inverse_multiquadric"}
                optional (default="gaussian")
        The type of RBF kernel to apply.
    """

    def __init__(
        self, in_features, out_features, centers=None, gamma=1.0, rbf_type="gaussian"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma
        self.rbf_type = rbf_type.lower()

        if centers is None:
            centers = torch.linspace(-1, 1, out_features).reshape(-1, 1)
            centers = centers.repeat(1, in_features)
        else:
            centers = torch.as_tensor(centers, dtype=torch.float32)

        self.centers = nn.Parameter(centers, requires_grad=True)

        valid_rbf_types = {"gaussian", "multiquadric", "inverse_multiquadric"}
        if self.rbf_type not in valid_rbf_types:
            raise ValueError(
                f"rbf_type must be one of {valid_rbf_types}, got {self.rbf_type}"
            )

    def forward(self, x):
        """Apply the RBF transformation to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor with RBF features.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)

        distances_squared = torch.sum(diff**2, dim=-1)

        if self.rbf_type == "gaussian":
            return torch.exp(-self.gamma * distances_squared)
        elif self.rbf_type == "multiquadric":
            return torch.sqrt(1 + self.gamma * distances_squared)
        else:  # inverse_multiquadric
            return 1 / torch.sqrt(1 + self.gamma * distances_squared)


class RBFNetwork(nn.Module):
    """Neural network with an RBF layer followed by fully connected layers.

    This model is designed to use RBF-transformed features as input for a series
    of linear transformations, enabling effective learning from non-linear
    representations.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of units in the RBF layer.
    output_size : int
        Number of output features for the network.
    centers : torch.Tensor, optional (default=None)
        Centers points for the RBF layer
    gamma : float, optional (default=1.0)
        Scaling factor controlling the spread of the RBF layer.
    rbf_type :{"gaussian", "multiquadric", "inverse_multiquadric"}
                optional (default="gaussian")
        Type of RBF kernel to apply.
    linear_layers : list of int, optional (default=[64, 32])
        Sizes of linear layers following the RBF layer
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        centers=None,
        gamma=1.0,
        rbf_type="gaussian",
        linear_layers=[64, 32],
    ):
        super().__init__()

        self.rbf_layer = RBFLayer(
            in_features=input_size,
            out_features=hidden_size,
            centers=centers,
            gamma=gamma,
            rbf_type=rbf_type,
        )

        layers = []
        prev_size = hidden_size

        for size in linear_layers:
            layers.extend([nn.Linear(prev_size, size), nn.ReLU(), nn.Dropout(0.1)])
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))
        self.sequential_layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x = self.rbf_layer(x)
        return self.sequential_layers(x)


class RBFForecaster(BaseDeepNetworkPyTorch):
    r"""Forecasting model using RBF transformations and 'NN' layers for time series.

    This forecaster uses an RBF layer to transform input time series data into a
    higher-dimensional space, which is then used by neural network layers for
    forecasting.

    The model structure is as follows:

    1. Input layer:
       - Takes in the input time series data of shape (batch_size, window_length)
    2. RBF Layer:
       - Applies Radial Basis Function (RBF) transformation to the input features
       - Uses centers and gamma parameter to control the spread of the RBFs
       - Expands the input feature space into a higher-dimensional representation
       - Output shape: (batch_size, hidden_size)
    3. Sequential layers:
       - One or more fully connected linear layers
       - Each linear layer uses the LazyLinear module, which initializes the weights
         based on the input size
       - ReLU activation is applied after each linear layer
       - Dropout with a rate of 0.1 is applied after each linear layer
       - The final linear layer produces the output prediction
       - Output shape: (batch_size, 1)

    Parameters
    ----------
    window_length : int
        Length of the input sequence for each window.
    hidden_size : int, optional (default=32)
        Number of units in the RBF layer.
    batch_size : int, optional (default=32)
        Size of mini-batches for training.
    centers : array-like, optional (default=None)
        Center points for RBF transformations.
    gamma : float, optional (default=1.0)
        Scaling factor controlling the spread of the RBF.
    rbf_type : {"gaussian", "multiquadric", "inverse_multiquadric"}
                optional (default="gaussian")
        Type of RBF kernel to use.
    linear_layers : list of int, optional (default=[64, 32])
        Sizes of linear layers following the RBF layer.
    optimizer : {"adam", "sgd", "rmsprop"}
                optional (default="adam")
        Type of optimizer to use.
    lr : float, optional (default=0.01)
        Learning rate for optimizer.
    epochs : int, optional (default=100)
        Number of training epochs.
    stride : int, optional (default=1)
        Step size between windows.
    criterion : {"mse", "l1", "poisson", "bce", "crossentropy"}
               optional (default="mse")
        Loss function to use during training.
    device : str, optional (default="cpu")
        Device to use for training and computation. Options are "cpu" or "cuda"
        for GPU computation if available.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["phoeenniixx"],
        "maintainers": ["phoeenniixx"],
        "python_dependencies": "torch",
        # estimator type
        # --------------
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": False,
    }

    def __init__(
        self,
        window_length=10,
        hidden_size=32,
        batch_size=32,
        centers=None,
        gamma=1.0,
        rbf_type="gaussian",
        linear_layers=[64, 32],
        optimizer="adam",
        lr=0.01,
        epochs=100,
        stride=1,
        criterion="mse",
        device="cpu",
    ):
        super().__init__()

        self._check_torch_availability()

        self.window_length = window_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.centers = centers
        self.gamma = gamma
        self.rbf_type = rbf_type
        self.linear_layers = linear_layers
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.stride = stride
        self.criterion = criterion

        if self._check_torch_availability():
            pass
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
        if isinstance(device, str):
            if device not in ["cpu", "cuda"]:
                raise ValueError("device must be 'cpu' or 'cuda'")
            if device == "cuda" and not torch.cuda.is_available():
                warn("CUDA is not available, using CPU instead", UserWarning)
                device = "cpu"
        return torch.device(device)

    def _check_torch_availability(self):
        """Check if torch is available and raise appropriate error if not."""
        if not _check_soft_dependencies("torch", severity="none"):
            raise ModuleNotFoundError(
                "torch is a required optional dependency for RBFForecaster. "
                "Please install torch to use this forecaster. "
                "Instructions can be found at https://pytorch.org/get-started/locally/"
            )
        return True

    def build_network(self, input_size):
        """Build the RBF network architecture."""
        output_size = 1

        self.network = RBFNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            output_size=output_size,
            centers=self.centers,
            gamma=self.gamma,
            rbf_type=self.rbf_type,
            linear_layers=self.linear_layers,
        ).to(self._device)

    def _fit(self, y, X=None, fh=None):
        """Fit the RBF-based model to the provided time series data.

        Parameters
        ----------
        y : pd.Series
            The time series data to be fitted.
        X : optional
            Additional exogenous data (not used).
        fh : optional
            Forecasting horizon (not used).
        """
        from torch.utils.data import DataLoader, TensorDataset

        self._y = y.copy()
        if isinstance(y, pd.Series):
            y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1))
        else:
            y_scaled = self.scaler.fit_transform(y.values)

        X_train, y_train = self._create_windows(y_scaled)
        X_tensor = torch.FloatTensor(X_train).to(self._device)
        y_tensor = torch.FloatTensor(y_train).to(self._device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_size = self.window_length
        self.build_network(input_size)

        self._criterion = self._instantiate_criterion()
        self._optimizer = self._instantiate_optimizer()

        self.network.train()
        for epoch in range(self.epochs):
            self._run_epoch(epoch, dataloader)

        return self

    def _predict(self, X=None, fh=None):
        """Generate predictions for the specified forecasting horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon for the predictions.
        X : optional
            Additional exogenous data (not used).
        """
        fh_abs = fh.to_absolute(self._y.index[-1]).to_numpy()
        n_steps = len(fh_abs)
        predictions = np.zeros(n_steps)

        last_window = self._y[-self.window_length :].copy()
        current_window = self.scaler.transform(last_window.values.reshape(-1, 1))

        self.network.eval()
        with torch.no_grad():
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

    def _create_windows(self, y):
        """Generate sliding windows from the time series data.

        Parameters
        ----------
        y : np.array
            Scaled time series data.

        Returns
        -------
        X : np.array
            Input windows for model training.
        y_out : np.array
            Target values for model training.
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = len(y)

        n_windows = max(0, (n_samples - self.window_length)) // self.stride + 1

        if n_windows < 1:
            raise ValueError(
                f"Not enough samples to create windows. Need at least "
                f"{self.window_length} samples, but got {n_samples}"
            )

        X = np.zeros((n_windows, self.window_length))
        y_out = np.zeros((n_windows, 1))

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_length

            if end_idx >= n_samples:
                break

            X[i] = y[start_idx:end_idx].flatten()

            y_out[i] = y[end_idx]

        return X[:i], y_out[:i]

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
        """Return testing parameter settings for RBFForecaster.

        Returns
        -------
        list
            List of parameter dictionaries for testing.
        """
        params_list = [
            {
                "window_length": 5,
                "hidden_size": 8,
                "batch_size": 16,
                "gamma": 1.0,
                "rbf_type": "gaussian",
                "linear_layers": [16],
                "epochs": 3,
                "lr": 0.01,
                "stride": 1,
                "optimizer": "adam",
                "device": "cpu",
            },
            {
                "window_length": 12,
                "hidden_size": 32,
                "batch_size": 32,
                "gamma": 0.5,
                "rbf_type": "gaussian",
                "linear_layers": [64, 32],
                "epochs": 10,
                "lr": 0.005,
                "stride": 2,
                "optimizer": "adam",
                "criterion": "mse",
                "device": "cuda",
            },
        ]

        return params_list
