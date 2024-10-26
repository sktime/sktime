"""RBF Neural Networks for Time Series Forecasting."""

__author__ = ["phoeenniixx"]

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sktime.forecasting.base import BaseForecaster
from sktime.transformations.series.basisfunction import RBFTransformer


def _check_dependencies():
    """Check if required dependencies are available."""
    from importlib.util import find_spec

    torch_available = find_spec("torch") is not None
    if not torch_available:
        raise ImportError(
            "PyTorch is required for RBFForecaster. "
            "Please install it with: pip install torch"
        )

    return torch_available


if _check_dependencies():
    import torch
    import torch.nn as nn
    from torch.optim import SGD, Adam, RMSprop
    from torch.utils.data import DataLoader, TensorDataset


class LazyLinear(nn.Module):
    r"""A lazy linear layer initialized based on the input dims during the forward pass.

    This layer is useful when the input size is dynamic or unknown at the initialization
    stage, enabling flexible architecture design.

    Parameters
    ----------
    out_features : int
        Number of output features for the linear layer.
    """

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.linear = None

    def forward(self, x):
        """Initialize and apply the linear layer if not already initialized.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be passed through the linear layer.

        Returns
        -------
        torch.Tensor
            Output tensor after linear transformation.
        """
        if self.linear is None:
            self.linear = nn.Linear(x.shape[-1], self.out_features).to(x.device)
        return self.linear(x)


class RBFLayer(nn.Module):
    r"""Radial Basis Function layer to transform input data into a new feature space.

    This layer applies an RBF transformation to each input feature,
    expanding the feature space based on distances to predefined center points.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features (or RBF centers).
    centers : array-like, optional (default=None)
        The centers :math:`c_k` for the RBF transformation.
        If None, centers are evenly spaced.
    gamma : float, optional (default=1.0)
        Parameter controlling the spread of the RBF.
    rbf_type : {"gaussian", "multiquadric", "inverse_multiquadric"}
                optional (default="gaussian")
        The type of RBF kernel to apply.
    """

    def __init__(
        self, in_features, out_features, centers=None, gamma=1.0, rbf_type="gaussian"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if centers is None:
            centers = torch.linspace(-1, 1, out_features).reshape(-1, 1)
            centers = centers.repeat(1, in_features).numpy()

        self.rbf_transformer = RBFTransformer(
            centers=centers,
            gamma=gamma,
            rbf_type=rbf_type,
            apply_to="values",
            use_torch=True,
        )

    def forward(self, x):
        """Apply the RBF transformation to the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor with RBF features.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_np = x.detach().cpu().numpy()
        self.rbf_transformer.fit(x_np)
        x_rbf = self.rbf_transformer.transform(x_np)
        x_rbf = torch.from_numpy(x_rbf).float().to(x.device)
        return x_rbf


class RBFNetwork(nn.Module):
    r"""Neural network with an RBF layer followed by fully connected layers.

    This model is designed to use RBF-transformed features as input for a series of
    linear transformations, enabling effective learning from non-linear representations.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of units in the RBF layer.
    output_size : int
        Number of output features for the network.
    centers : array-like, optional (default=None)
        Center points for the RBF layer.
    gamma : float, optional (default=1.0)
        Scaling factor controlling the spread of the RBF layer.
    rbf_type : {"gaussian", "multiquadric", "inverse_multiquadric"}
                optional (default="gaussian")
        Type of RBF kernel to apply.
    linear_layers : list of int, optional (default=[64, 32])
        Sizes of linear layers following the RBF layer.
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
        for linear_size in linear_layers:
            layers.extend([LazyLinear(linear_size), nn.ReLU(), nn.Dropout(0.1)])

        layers.append(LazyLinear(output_size))
        self.sequential_layers = nn.Sequential(*layers)

    def forward(self, x):
        """Pass input data through the RBF and sequential layers.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through RBF and linear layers.
        """
        x = self.rbf_layer(x)
        x = self.sequential_layers(x)
        return x


class RBFForecaster(BaseForecaster):
    r"""
    Forecasting model using RBF transformations and 'NN' layers for time series.

    This forecaster uses an RBF layer to transform input time series data into a
    higher-dimensional space, which is then used by neural network layers for
    forecasting.

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
    loss_fn : {"mse", "l1", "poisson", "bce", "crossentropy"}
               optional (default="mse")
        Loss function to use during training.
    use_cuda : bool, optional (default=False)
        Whether to use GPU for training if available.
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
        window_length,
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
        loss_fn="mse",
        use_cuda=False,
    ):
        # Check PyTorch availability during initialization
        _check_dependencies()

        super().__init__()
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
        self.loss_fn = loss_fn
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self._is_fitted = False
        self.model = None
        self.scaler = StandardScaler()

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
        self._y = y.copy()
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1))
        X_train, y_train = self._create_windows(y_scaled)

        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = RBFNetwork(
            input_size=self.window_length,
            hidden_size=self.hidden_size,
            output_size=1,
            centers=self.centers,
            gamma=self.gamma,
            rbf_type=self.rbf_type,
            linear_layers=self.linear_layers,
        ).to(self.device)

        criterion = self._create_loss_fn()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer = self._create_optimizer()
                optimizer.zero_grad()
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader)}")

        self._is_fitted = True
        return self

    def _predict(self, fh, X=None):
        """Generate predictions for the specified forecasting horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon for the predictions.
        X : optional
            Additional exogenous data (not used).
        """
        y_scaled = self.scaler.transform(self._y.values.reshape(-1, 1))
        X_pred, _ = self._create_windows(y_scaled)
        X_pred_tensor = torch.FloatTensor(X_pred).to(self.device)
        y_pred_tensor = self.model(X_pred_tensor)
        y_pred = y_pred_tensor.detach().cpu().numpy()
        y_pred = self.scaler.inverse_transform(y_pred)
        return pd.Series(y_pred.flatten(), index=self._y.index[-len(fh) :])

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
        n_windows = (len(y) - self.window_length) // self.stride + 1
        if n_windows < 1:
            raise ValueError(
                "Not enough samples to create windows. "
                f"Need at least {self.window_length + 1} samples."
            )
        X = np.zeros((n_windows, self.window_length))
        y_out = np.zeros((n_windows, 1))
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_length
            X[i] = y[start_idx:end_idx].flatten()
            y_out[i] = y[end_idx]
        return X, y_out

    def _create_optimizer(self):
        """Initialize optimizer based on the specified optimizer type.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer instance.
        """
        optimizers = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
        if self.optimizer.lower() not in optimizers:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        return optimizers[self.optimizer.lower()](
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        )

    def _create_loss_fn(self):
        """Initialize loss function based on the specified type.

        Returns
        -------
        nn.Module
            Loss function instance.
        """
        loss_fns = {
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "poisson": nn.PoissonNLLLoss,
            "bce": nn.BCELoss,
            "crossentropy": nn.CrossEntropyLoss,
        }
        if self.loss_fn.lower() not in loss_fns:
            raise ValueError(f"Unsupported Loss Function: {self.loss_fn}")
        return loss_fns[self.loss_fn.lower()]()

    def plot_predictions(self, y_train, y_test, y_pred):
        """Plot actual and predicted values for time series data.

        Parameters
        ----------
        y_train : pd.Series
            Training time series data.
        y_test : pd.Series
            True values for testing data.
        y_pred : pd.Series
            Model predictions for testing data.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting predictions. "
                "Please install it with: pip install matplotlib"
            )

        y_train, y_test, y_pred = (
            s.to_timestamp()
            if hasattr(s.index, "dtype") and str(s.index.dtype).startswith("period")
            else s
            for s in (y_train, y_test, y_pred)
        )
        plt.figure(figsize=(12, 6))
        plt.plot(y_train.index, y_train.values, label="Training Data", color="blue")
        plt.plot(y_test.index, y_test.values, label="Actual Values", color="green")
        plt.plot(
            y_pred.index,
            y_pred.values,
            label="Predictions",
            color="red",
            linestyle="--",
        )
        plt.title("Time Series Forecast")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    @classmethod
    def get_test_params(cls):
        """Return test parameter sets for the transformer.

        Returns
        -------
        list
            List of parameter dictionaries for testing.
        """
        return [
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
                "use_cuda": False,
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
                "loss_fn": "mse",
                "use_cuda": True,
            },
        ]
