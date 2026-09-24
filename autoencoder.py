import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sktime.base import BaseEstimator
# from sktime.annotation.base import BaseSeriesAnnotator


class AutoEncoder(BaseEstimator):
    """Autoencoder-based anomaly detector for time series.

    This estimator uses a feedforward neural network trained to reconstruct
    input data. Anomaly scores are computed as reconstruction error.

    Parameters
    ----------
    contamination : float, default=0.1
        The amount of contamination of the data set, 
        i.e. the proportion of outliers in the data set.

    lr : float, default=1e-3
        The learning rate for optimizer.

    epoch_num : int, default=10
        The number of training epochs.

    batch_size : int, default=32
        The batch size for training.

    hidden_neuron_list : list of int, optional (default=[64, 32])
        The number of neurons in hidden layers.

    activation : str, default="relu"
        The activation function ("relu", "tanh", "sigmoid").

    batch_norm : bool, default=True
        Whether to use batch normalization.

    dropout_rate : float, default=0.2
        Dropout rate.

    device : str, optional
        Device to use ("cpu" or "cuda"). If None, auto-detect.

    random_state : int, default=42
        Random seed for reproducibility.

    verbose : int, default=1
        Verbosity level.
    """

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "univariate-only": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        contamination=0.1,
        lr=1e-3,
        epoch_num=10,
        batch_size=32,
        hidden_neuron_list=None,
        activation="relu",
        batch_norm=True,
        dropout_rate=0.2,
        device=None,
        random_state=42,
        verbose=1,
    ):
        self.contamination = contamination
        self.lr = lr
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.hidden_neuron_list = hidden_neuron_list or [64, 32]
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        super().__init__()

    def fit(self, X, y=None):
        """Fit the autoencoder model. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Training data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame()

        X_np = X.values.astype(np.float32)
        self.feature_size_ = X_np.shape[1]

        self.device_ = self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        torch.manual_seed(self.random_state)

        self.model_ = AutoEncoderModel(
            feature_size=self.feature_size_,
            hidden_neuron_list=self.hidden_neuron_list,
            activation=self.activation,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
        ).to(self.device_)

        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        self.criterion_ = nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X_np))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()

        for epoch in range(self.epoch_num):
            epoch_loss = 0.0

            for (batch,) in loader:
                batch = batch.to(self.device_)

                self.optimizer_.zero_grad()
                recon = self.model_(batch)
                loss = self.criterion_(recon, batch)

                loss.backward()
                self.optimizer_.step()

                epoch_loss += loss.item()

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epoch_num}, Loss: {epoch_loss:.4f}")

        scores = self._compute_scores(X_np)
        self.threshold_ = np.percentile(
            scores, 100 * (1 - self.contamination)
        )

        return self

    def predict_scores(self, X):
        """Compute anomaly scores using reconstruction error.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series

        Returns
        -------
        scores : pd.Series
            Reconstruction error for each sample.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame()

        X_np = X.values.astype(np.float32)
        scores = self._compute_scores(X_np)

        return pd.Series(scores, index=X.index)

    def predict(self, X):
        """Predict binary anomaly labels.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series

        Returns
        -------
        labels : pd.Series
            1 indicates anomaly, 0 indicates normal.
        """
        scores = self.predict_scores(X)
        labels = (scores > self.threshold_).astype(int)
        return labels

    def _compute_scores(self, X_np):
        """Compute reconstruction error."""
        self.model_.eval()

        X_tensor = torch.from_numpy(X_np).to(self.device_)

        with torch.no_grad():
            recon = self.model_(X_tensor)

        errors = torch.mean((X_tensor - recon) ** 2, dim=1)
        return errors.cpu().numpy()


class AutoEncoderModel(nn.Module):
    """Feedforward autoencoder neural network."""

    def __init__(
        self,
        feature_size,
        hidden_neuron_list,
        activation="relu",
        batch_norm=True,
        dropout_rate=0.2,
    ):
        super().__init__()

        self.encoder = self._build_encoder(
            feature_size, hidden_neuron_list, activation, batch_norm, dropout_rate
        )

        self.decoder = self._build_decoder(
            feature_size, hidden_neuron_list, activation, batch_norm, dropout_rate
        )

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def _build_encoder(self, input_dim, hidden_layers, activation, batch_norm, dropout):
        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        return nn.Sequential(*layers)

    def _build_decoder(self, output_dim, hidden_layers, activation, batch_norm, dropout):
        layers = []
        hidden_layers = list(reversed(hidden_layers))
        prev_dim = hidden_layers[0]

        for h in hidden_layers[1:]:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))