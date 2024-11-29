"""RBF Neural Networks for Time Series Forecasting."""

__author__ = ["phoeenniixx"]

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sktime.forecasting.base.adapters._pytorch import BaseDeepNetworkPyTorch
from sktime.transformations.series.basisfunction import RBFTransformer
from sktime.utils.dependencies._dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="warning"):
    import torch
    import torch.nn as nn

else:
    class nn:
        """dummy class"""


class LazyLinear(nn.Module):
    r"""A lazy linear layer initialized based on the input dims.

    This layer is useful when the input size is dynamic or unknown at the
    initialization stage, enabling flexible architecture design.

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
    r"""RBF layer to transform input data into a new feature space.

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
            self,
            in_features,
            out_features,
            centers=None,
            gamma=1.0,
            rbf_type="gaussian",
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

    This model is designed to use RBF-transformed features as input for a series
    of linear transformations, enabling effective learning from non-linear
    representations.

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
        criterion="mse",
        use_cuda=False,
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

        self.use_cuda = use_cuda

        if self._check_torch_availability():
            import torch

        self._cuda_available = torch.cuda.is_available()
        self._use_cuda_actual = self.use_cuda and self._cuda_available
        self.device = torch.device("cuda" if self._use_cuda_actual else "cpu")

        self.model = None
        self.scaler = StandardScaler()

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

        self.model = RBFNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            output_size=output_size,
            centers=self.centers,
            gamma=self.gamma,
            rbf_type=self.rbf_type,
            linear_layers=self.linear_layers,
        ).to(self.device)

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
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._y = y.copy()
        if isinstance(y, pd.Series):
            y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1))
        else:
            y_scaled = self.scaler.fit_transform(y.values)

        X_train, y_train = self._create_windows(y_scaled)
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_size = self.window_length
        self.build_network(input_size)

        criterion = self._instantiate_criterion()
        optimizer = None

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                if optimizer is None:
                    _ = self.model(batch_X)
                    optimizer = self._instantiate_optimizer()

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

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
        import torch

        fh_abs = fh.to_absolute(self._y.index[-1]).to_numpy()
        n_steps = len(fh_abs)
        predictions = np.zeros(n_steps)

        last_window = self._y[-self.window_length :].copy()
        current_window = self.scaler.transform(last_window.values.reshape(-1, 1))

        self.model.eval()
        with torch.no_grad():
            for i in range(n_steps):
                X_predict = torch.FloatTensor(current_window.reshape(1, -1)).to(
                    self.device
                )
                pred_scaled = self.model(X_predict).cpu().numpy()
                pred = self.scaler.inverse_transform(pred_scaled.reshape(1, -1))
                predictions[i] = pred.ravel()[0]

                current_window = np.roll(current_window, -1, axis=0)
                current_window[-1] = pred_scaled.ravel()[0]

        return pd.Series(predictions, index=fh_abs, name=self._y.name)


    def plot_predictions(self, y_train, y_test, y_pred):
        """Plot the training data, actual values, and predictions."""
        import matplotlib.pyplot as plt

        if hasattr(y_train.index, "dtype") and str(y_train.index.dtype).startswith(
            "period"
        ):
            y_train.index = y_train.index.to_timestamp()
        if hasattr(y_test.index, "dtype") and str(y_test.index.dtype).startswith(
            "period"
        ):
            y_test.index = y_test.index.to_timestamp()
        if hasattr(y_pred.index, "dtype") and str(y_pred.index.dtype).startswith(
            "period"
        ):
            y_pred.index = y_pred.index.to_timestamp()

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
        return optimizers[self.optimizer.lower()](self.model.parameters(), lr=self.lr)

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
                "criterion": "mse",
                "use_cuda": True,
            },
        ]

        return params_list
