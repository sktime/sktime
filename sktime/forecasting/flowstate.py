"""Granite FlowState Forecaster for time series forecasting.

This module implements the GraniteFlowStateForecaster, a deep learning-based
time series forecasting model built on a State Space Model (SSM) encoder and
a polynomial Functional Basis Decoder.

Architecture Overview
---------------------
1. Causal Normalization : Online normalization that only uses past statistics,
   preventing look-ahead bias during training.
2. SSMBlock : A recurrent state-space layer that propagates a latent state
   through time using learnable transition matrices (A, B, C, D).
3. SSMEncoderLayer : Wraps SSMBlock with a residual connection and an MLP
   for non-linear feature transformation.
4. FunctionalBasisDecoder : Projects the final encoder hidden state onto a
   polynomial basis to produce multi-step forecasts in a single forward pass.
5. FlowStateModel : End-to-end model that chains the above components.
6. GraniteFlowStateForecaster : sktime-compatible estimator wrapping
   FlowStateModel, exposing fit and predict methods.
"""

__author__ = ["FlyingDragon112"]

import numpy as np

from sktime.forecasting.base.adapters import _pytorch
from sktime.utils.dependencies import _safe_import

# Lazy imports so the module can be loaded without PyTorch installed.
# _safe_import returns a placeholder that raises an informative error on use.
torch = _safe_import("torch")
nn = _safe_import("torch.nn")
F = _safe_import("torch.nn.functional")


def causal_normalize(x):
    """Normalize a sequence causally using running mean and variance.

    At each time step t, only the statistics of the sub-sequence
    x[:, :t+1, :] are used — no future information leaks into the
    normalization, making this safe for autoregressive training.

    Parameters
    ----------
    x : torch.Tensor, shape (batch, seq_len, channels)
        Raw input sequence.

    Returns
    -------
    x_norm : torch.Tensor, shape (batch, seq_len, channels)
        Causally normalized sequence with zero mean and unit variance at
        each step.
    mean_last : torch.Tensor, shape (batch, channels)
        Running mean evaluated at the last time step.  Used to
        de-normalize forecasts.
    std_last : torch.Tensor, shape (batch, channels)
        Running standard deviation at the last time step (with a small
        epsilon for numerical stability).  Used to de-normalize forecasts.
    """
    cumsum = torch.cumsum(x, dim=1)
    steps = torch.arange(1, x.size(1) + 1, device=x.device).view(1, -1, 1)
    mean = cumsum / steps
    var = torch.cumsum((x - mean) ** 2, dim=1) / steps
    std = torch.sqrt(var + 1e-6)
    x_norm = (x - mean) / std

    return x_norm, mean[:, -1], std[:, -1]


class SSMBlock(nn.Module):
    """Linear State Space Model (SSM) layer.

    Implements a discrete-time linear dynamical system:

        h_t = h_{t-1} @ A + x_t @ B
        y_t =     h_t @ C + x_t @ D

    where h_t is the hidden state and x_t is the input at step t.
    All four system matrices (A, B, C, D) are learned parameters.

    Parameters
    ----------
    d_model : int
        Dimensionality of both the input features and the hidden state.
        The state space is therefore d_model-dimensional.
    """

    def __init__(self, d_model):
        super().__init__()

        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.D = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def forward(self, x):
        """Run the SSM recurrence over the full input sequence.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, d_model)
            Embedded input sequence.

        Returns
        -------
        torch.Tensor, shape (batch, seq_len, d_model)
            Output sequence produced by the SSM at every time step.
        """
        batch, seq, dim = x.shape

        state = torch.zeros(batch, dim, device=x.device)

        outputs = []
        for t in range(seq):
            xt = x[:, t]
            state = state @ self.A + xt @ self.B
            out = state @ self.C + xt @ self.D
            outputs.append(out)

        return torch.stack(outputs, dim=1)


class SSMEncoderLayer(nn.Module):
    """Single encoder block combining an SSM and a position-wise MLP.

    The layer applies two residual sub-layers:

    1. x ← x + SSMBlock(x) — captures sequential dependencies.
    2. x ← x + MLP(x)      — applies non-linear feature mixing.

    The MLP follows the standard Transformer feed-forward design with a
    4x expansion factor and GELU activations.

    Parameters
    ----------
    dim : int
        Feature dimensionality (d_model).  Used for both the SSM and
        the MLP (which expands to 4 * dim internally).
    """

    def __init__(self, dim):
        super().__init__()

        self.ssm = SSMBlock(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        """Apply one encoder layer to the input sequence.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, dim)
            Input feature sequence.

        Returns
        -------
        torch.Tensor, shape (batch, seq_len, dim)
            Transformed feature sequence after SSM and MLP residual blocks.
        """
        h = self.ssm(x)
        x = x + h
        x = x + self.mlp(x)

        return x


class FunctionalBasisDecoder(nn.Module):
    """Decode a coefficient vector into a forecast using a polynomial basis.

    Given a coefficient vector c ∈ R^{hidden_dim}, the decoder produces
    a forecast of length horizon as a linear combination of monomials:

        ŷ(t) = Σ_{i=0}^{hidden_dim-1}  c_i · t^i,   t ∈ [-1, 1]

    This parameterises the forecast as a polynomial of degree
    hidden_dim - 1, enabling smooth, structured predictions from a
    compact latent representation.

    Parameters
    ----------
    hidden_dim : int
        Number of polynomial basis functions (= degree + 1).
        Must match the encoder's output dimensionality.
    horizon : int
        Number of future time steps to forecast.
    """

    def __init__(self, hidden_dim, horizon):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.horizon = horizon

    def forward(self, coeff):
        """Evaluate the polynomial forecast from coefficient vector.

        Parameters
        ----------
        coeff : torch.Tensor, shape (batch, hidden_dim)
            Polynomial coefficients produced by the encoder.

        Returns
        -------
        torch.Tensor, shape (batch, horizon, 1)
            Forecast values at each of the horizon future time steps.
        """
        device = coeff.device

        t = torch.linspace(-1, 1, self.horizon, device=device)
        basis = []
        for i in range(self.hidden_dim):
            basis.append(torch.pow(t, i))

        basis = torch.stack(basis)
        y = coeff @ basis

        return y.unsqueeze(-1)


class FlowStateModel(nn.Module):
    """End-to-end FlowState forecasting network.

    The forward pass consists of four stages:

    1. Causal normalization — removes non-stationary trends causally.
    2. Input embedding — projects raw (normalized) features to
       hidden_dim via a learned linear layer.
    3. SSM Encoder — a stack of layers SSMEncoderLayers that refine
       the sequence representation.
    4. Functional Basis Decoder — the last token's hidden state is used
       as polynomial coefficients and evaluated over the forecast horizon.
    5. De-normalization — the forecast is rescaled back to the original
       value range using the statistics saved in step 1.

    Parameters
    ----------
    input_dim : int
        Number of input channels (1 for univariate, >1 for multivariate).
    hidden_dim : int
        Internal feature dimensionality used throughout the network.
    layers : int
        Number of stacked SSMEncoderLayers.
    horizon : int
        Number of future time steps to predict.

    References
    ----------
    .. [1] Lars Graf, Thomas Ortner, Stanisław Woźniak, Angeliki Pantazi
           (2025). FlowState: Sampling Rate Invariant Time Series
           Forecasting. https://arxiv.org/abs/2508.05287
    """

    _tags = {
        "authors": ["FlyingDragon112"],
        "maintainers": ["FlyingDragon112"],
        "fit_is_empty": False,
        "capability:update": False,
        "capability:exogenous": False,
    }

    def __init__(self, input_dim, hidden_dim, layers, horizon):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.Sequential(
            *[SSMEncoderLayer(hidden_dim) for _ in range(layers)]
        )
        self.decoder = FunctionalBasisDecoder(hidden_dim, horizon)

    def forward(self, x):
        """Perform a full forward pass and return the de-normalized forecast.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, context_len, input_dim)
            Historical (context) sequence fed to the model.

        Returns
        -------
        torch.Tensor, shape (batch, horizon, 1)
            Point forecast for the next horizon time steps in the
            original (de-normalized) value scale.
        """
        x, mean, std = causal_normalize(x)
        x = self.embedding(x)
        x = self.encoder(x)
        coeff = x[:, -1]
        forecast = self.decoder(coeff)
        forecast = forecast * std.unsqueeze(1) + mean.unsqueeze(1)

        return forecast


class GraniteFlowStateForecaster(_pytorch.BaseDeepNetworkPyTorch):
    """Granite FlowState Time Series Forecaster.

    A deep learning forecaster that combines a causal State Space Model (SSM)
    encoder with a polynomial Functional Basis Decoder.  The model is designed
    for univariate time series and is trained end-to-end to minimize mean
    squared error on a fixed-length forecast horizon.

    The forecaster follows the sktime BaseForecaster API and inherits
    training-loop utilities from BaseDeepNetworkPyTorch.

    Parameters
    ----------
    input_dim : int, default=1
        Number of input channels.  Use 1 for univariate series.
    hidden_dim : int, default=64
        Dimensionality of the SSM hidden state and MLP layers.
        Larger values increase model capacity but also memory and compute cost.
    layers : int, default=3
        Number of stacked SSMEncoderLayer blocks in the encoder.
    horizon : int, default=24
        Forecast horizon — the number of future steps to predict.
        Must match the fh argument passed to predict.
    lr : float, default=1e-3
        Learning rate for the Adam optimizer.
    epochs : int, default=10
        Number of full passes over the training sequence.
    device : str, default="cpu"
        PyTorch device string (e.g. "cpu","cuda", "mps").

    Attributes
    ----------
    model : FlowStateModel
        The underlying PyTorch network, available after calling fit.
    last_window : torch.Tensor
        The full training sequence tensor stored for use in predict and
        update.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.forecasting.granite_flowstate import GraniteFlowStateForecaster
    >>> y = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 100)))
    >>> forecaster = GraniteFlowStateForecaster(
    ...     hidden_dim=32, layers=2, horizon=12, epochs=5
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    GraniteFlowStateForecaster(...)
    >>> y_pred = forecaster.predict(fh=list(range(1, 13)))  # doctest: +SKIP
    """

    _tags = {
        "authors": ["FlyingDragon112"],
        "maintainers": ["FlyingDragon112"],
    }

    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        layers=3,
        horizon=24,
        lr=1e-3,
        epochs=10,
        device="cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.horizon = horizon
        self.lr = lr
        self.epochs = epochs
        self.device = device

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the FlowState model to training data.

        Constructs a single context-target split from y:
        * Context : y[:-horizon]
        * Target  : y[-horizon:]

        The model is trained for self.epochs gradient steps using Adam
        and MSE loss.

        Parameters
        ----------
        y : array-like, shape (n_timepoints,)
            Training time series values.
        X : ignored
            Exogenous variables are not supported; this parameter exists only
            for API compatibility.
        fh : ignored
            Forecast horizon is set at construction time via self.horizon.

        Returns
        -------
        self : GraniteFlowStateForecaster
            Fitted estimator.
        """
        y = np.asarray(y).astype(np.float32)

        seq = torch.tensor(y, dtype=torch.float32).view(1, -1, 1).to(self.device)

        self.model = FlowStateModel(
            self.input_dim,
            self.hidden_dim,
            self.layers,
            self.horizon,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            self.model.train()

            context = seq[:, : -self.horizon]
            target = seq[:, -self.horizon :]

            pred = self.model(context)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.last_window = seq

        return self

    def _predict(self, y=None, X=None, fh=None):
        """Generate point forecasts for the next horizon time steps.

        Uses the context portion of self.last_window (i.e. everything
        except the last horizon steps) to produce forecasts.

        Parameters
        ----------
        y : ignored
        X : ignored
        fh : ignored
            Forecast horizon is fixed at self.horizon.

        Returns
        -------
        pred : np.ndarray, shape (horizon,)
            Flat array of point forecasts.
        """
        self.model.eval()

        context = self.last_window[:, : -self.horizon]

        with torch.no_grad():
            pred = self.model(context)

        pred = pred.cpu().numpy().flatten()

        return pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return lightweight parameter configurations for unit tests.

        Parameters
        ----------
        parameter_set : str, default="default"
            Identifier for the parameter set.  Currently only "default"
            is supported.

        Returns
        -------
        list of dict
            A list containing one parameter dictionary.  The settings use a
            small model (hidden_dim=32, layers=2) and very few epochs
            (epochs=2) to keep test runtime minimal.
        """
        return [
            {
                "hidden_dim": 32,
                "layers": 2,
                "epochs": 2,
                "horizon": 12,
            }
        ]
