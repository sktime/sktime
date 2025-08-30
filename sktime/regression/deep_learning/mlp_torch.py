"""Multi Layer Perceptron Network (MLP) for regression using PyTorch."""

__author__ = ["Jack Russon"]
__all__ = ["MLPTorchRegressor"]

import numpy as np

from sktime.regression.deep_learning.base import BaseDeepRegressor
from sktime.networks.mlp_torch import PyTorchMLPNetwork
from sktime.utils.dependencies import _check_dl_dependencies

if _check_dl_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn


class MLPTorchRegressor(BaseDeepRegressor):
    """Multi Layer Perceptron Network (MLP) for regression using PyTorch.
    
    Parameters
    ----------
    n_epochs : int, default=2000
        The number of epochs to train the model.
    batch_size : int, default=16
        The number of samples per gradient update.
    random_state : int, optional, default=None
        Seed for random number generation.
    verbose : bool, default=False
        Whether to output extra information during training.
    loss : str, default="mean_squared_error"
        Loss function to use for regression.
    activation : str, default="relu"
        Activation function to use in hidden layers.
    hidden_dims : list, default=[500, 500, 500]
        List of hidden layer dimensions.
    dropout : float, default=0.0
        Dropout rate for regularization.
    use_bias : bool, default=True
        Whether to use bias in linear layers.
    optimizer : str, optional, default=None
        Optimizer to use. If None, uses Adam.
    lr : float, default=0.001
        Learning rate for the optimizer.
    """

    _tags = {
        "authors": ["Jack Russon"],
        "maintainers": ["Jack Russon"],
        "python_dependencies": ["torch"],
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
    }

    def __init__(
        self,
        n_epochs=2000,
        batch_size=16,
        random_state=None,
        verbose=False,
        loss="mean_squared_error",
        activation="relu",
        hidden_dims=None,
        dropout=0.0,
        use_bias=True,
        optimizer=None,
        lr=0.001,
    ):
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [500, 500, 500]
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.loss = loss
        self.activation = activation
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_bias = use_bias
        self.optimizer = optimizer
        self.lr = lr
        
        # Set random state for PyTorch
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        super().__init__()
        
        self._network = None
        self._criterion = None
        self._optimizer = None
        self.history = None

    def _internal_convert(self, X, y=None):
        """Override to enforce strict 3D input validation for PyTorch regressors.
        
        PyTorch regressors require 3D input and we don't allow automatic conversion 
        from 2D to 3D as this can mask user errors and lead to unexpected behavior.
        """
        if isinstance(X, np.ndarray) and X.ndim != 3:
            raise ValueError(
                f"Expected 3D input X with shape (n_instances, n_dims, series_length), "
                f"but got shape {X.shape}. PyTorch regressors require properly formatted "
                f"3D time series data. Please reshape your data or use a supported Panel mtype."
            )
        
        # Call parent method for other conversions
        return super()._internal_convert(X, y)

    def build_model(self, input_shape, **kwargs):
        """Build the PyTorch MLP network for regression.
        
        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
            
        Returns
        -------
        PyTorchMLPNetwork
            The built PyTorch network
        """
        # For regression, we need 1 output (continuous value)
        # We'll modify the network to output 1 value instead of num_classes
        return PyTorchMLPNetwork(
            input_shape, 
            num_classes=1,  # 1 output for regression
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_bias=self.use_bias
        )

    def _fit(self, X, y):
        """Fit the PyTorch MLP regressor.
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_instances, n_dims, series_length)
        y : np.ndarray
            Target values of shape (n_instances,)
        """
        # Build the network
        self._network = self.build_model(X.shape[1:])
        
        # Set up loss function
        if self.loss == "mean_squared_error":
            self._criterion = nn.MSELoss()
        elif self.loss == "mean_absolute_error":
            self._criterion = nn.L1Loss()
        elif self.loss == "huber":
            self._criterion = nn.HuberLoss()
        else:
            self._criterion = nn.MSELoss()
        
        # Set up optimizer
        if self.optimizer == "Adam":
            self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            self._optimizer = torch.optim.SGD(self._network.parameters(), lr=self.lr)
        elif self.optimizer == "AdamW":
            self._optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.lr)
        else:
            self._optimizer = torch.optim.Adam(self._network.parameters(), lr=self.lr)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Add dimension for batch
        
        # Training loop
        self._network.train()
        losses = []
        
        for epoch in range(self.n_epochs):
            # Forward pass
            y_pred = self._network(X_tensor)
            loss = self._criterion(y_pred, y_tensor)
            
            # Backward pass
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            
            losses.append(loss.item())
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}: Loss: {loss.item():.6f}")
        
        # Store training history
        self.history = {"loss": losses}

    def _predict(self, X):
        """Predict regression values for X.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_instances, n_dims, series_length)
            
        Returns
        -------
        np.ndarray
            Predicted values of shape (n_instances,)
        """
        if self._network is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Set to evaluation mode
        self._network.eval()
        
        with torch.no_grad():
            y_pred = self._network(X_tensor)
        
        # Convert back to numpy and squeeze
        return y_pred.numpy().squeeze()

    def summary(self):
        """Return training history summary.
        
        Returns
        -------
        dict or None
            Dictionary containing training losses if available
        """
        return self.history
    
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class
        """
        params = [
            {
                "n_epochs": 3,
                "batch_size": 4,
                "hidden_dims": [100, 50],
                "activation": "relu",
                "dropout": 0.0,
                "use_bias": True,
                "loss": "mean_squared_error",
                "random_state": 42,
            },
            {
                "n_epochs": 2,
                "batch_size": 2,
                "hidden_dims": [64],
                "activation": "tanh",
                "dropout": 0.1,
                "use_bias": False,
                "loss": "mean_absolute_error",
                "optimizer": "AdamW",
                "lr": 0.01,
                "random_state": 0,
            },
        ]
        return params
