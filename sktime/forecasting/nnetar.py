# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Neural network autoregression forecaster."""

__author__ = ["Saithej2k"]
__all__ = ["NNetAR"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class NNetAR(BaseForecaster):
    """Neural network autoregression forecaster.

    NNetAR fits feed-forward neural networks with one hidden layer to lagged
    values of the target series. Multi-step forecasts are produced recursively,
    and forecasts from repeated random initializations are averaged.

    This implements the core univariate autoregressive part of the ``nnetar``
    algorithm from the R ``forecast`` package. Exogenous regressors and
    simulation-based prediction intervals are not implemented.

    Parameters
    ----------
    p : int, default=1
        Number of non-seasonal lags to use as inputs.
    P : int, default=1
        Number of seasonal lags to use as inputs.
    sp : int, default=1
        Seasonal periodicity. Seasonal lag inputs are ``sp, 2 * sp, ..., P * sp``.
    size : int or None, default=None
        Number of hidden units. If None, uses half the number of inputs plus one.
    repeats : int, default=20
        Number of networks to fit with different random initial weights.
    scale_inputs : bool, default=True
        If True, standardize lagged inputs before fitting the neural networks.
    activation : {"identity", "logistic", "tanh", "relu"}, default="relu"
        Activation function for the hidden layer.
    solver : {"lbfgs", "sgd", "adam"}, default="lbfgs"
        Solver for ``sklearn.neural_network.MLPRegressor``.
    alpha : float, default=0.0001
        L2 regularization strength passed to ``MLPRegressor``.
    learning_rate_init : float, default=0.001
        Initial learning rate passed to ``MLPRegressor``.
    max_iter : int, default=200
        Maximum optimizer iterations passed to ``MLPRegressor``.
    tol : float, default=1e-4
        Optimization tolerance passed to ``MLPRegressor``.
    random_state : int, RandomState instance or None, default=None
        Controls random seeds used for the repeated neural network fits.

    Attributes
    ----------
    lags_ : np.ndarray of int
        Lag offsets used as model inputs.
    models_ : list
        Fitted ``MLPRegressor`` models.
    scaler_ : StandardScaler or None
        Fitted input scaler when ``scale_inputs=True``.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.nnetar import NNetAR
    >>> y = load_airline()
    >>> forecaster = NNetAR(p=3, P=1, sp=12, repeats=2, random_state=0)
    >>> forecaster.fit(y)
    NNetAR(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        "authors": ["Saithej2k"],
        "maintainers": "sktime developers",
        "requires-fh-in-fit": False,
        "capability:exogenous": False,
        "capability:insample": False,
        "capability:missing_values": False,
        "capability:random_state": True,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
    }

    def __init__(
        self,
        p=1,
        P=1,
        sp=1,
        size=None,
        repeats=20,
        scale_inputs=True,
        activation="relu",
        solver="lbfgs",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=200,
        tol=1e-4,
        random_state=None,
    ):
        self.p = p
        self.P = P
        self.sp = sp
        self.size = size
        self.repeats = repeats
        self.scale_inputs = scale_inputs
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit NNetAR to training data."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils import check_random_state

        self._validate_params()

        y_values = y.to_numpy(dtype=float)
        self.lags_ = self._make_lags()
        X_train, y_train = self._make_training_data(y_values)

        if len(y_train) == 0:
            raise ValueError(
                "NNetAR requires at least one complete training window after "
                "applying the configured lags."
            )

        self.scaler_ = None
        if self.scale_inputs:
            self.scaler_ = StandardScaler()
            X_train = self.scaler_.fit_transform(X_train)

        n_inputs = len(self.lags_)
        self.size_ = self.size if self.size is not None else n_inputs // 2 + 1
        rng = check_random_state(self.random_state)
        self.models_ = []

        for _ in range(self.repeats):
            model = MLPRegressor(
                hidden_layer_sizes=(self.size_,),
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=rng.randint(np.iinfo(np.int32).max),
            )
            model.fit(X_train, y_train)
            self.models_.append(model)

        self._fit_values_ = y_values
        return self

    def _predict(self, fh, X=None):
        """Forecast at the passed forecasting horizon."""
        fh_rel = fh.to_relative(self.cutoff).to_numpy()
        fh_steps = fh_rel.astype(int)

        if np.any(fh_steps <= 0):
            raise NotImplementedError("NNetAR supports out-of-sample forecasts only.")

        max_step = int(np.max(fh_steps))
        predictions = self._recursive_predict(max_step)
        y_pred = np.asarray(predictions)[fh_steps - 1]

        index = fh.to_absolute_index(self.cutoff)
        return pd.Series(y_pred, index=index, name=self._y.name)

    def _validate_params(self):
        """Validate constructor parameters."""
        if not isinstance(self.p, (int, np.integer)) or self.p < 0:
            raise ValueError("p must be a non-negative integer.")
        if not isinstance(self.P, (int, np.integer)) or self.P < 0:
            raise ValueError("P must be a non-negative integer.")
        if not isinstance(self.sp, (int, np.integer)) or self.sp < 1:
            raise ValueError("sp must be a positive integer.")
        if self.size is not None and (
            not isinstance(self.size, (int, np.integer)) or self.size < 1
        ):
            raise ValueError("size must be None or a positive integer.")
        if not isinstance(self.repeats, (int, np.integer)) or self.repeats < 1:
            raise ValueError("repeats must be a positive integer.")

    def _make_lags(self):
        """Return sorted non-seasonal and seasonal lag offsets."""
        lags = list(range(1, self.p + 1))
        lags.extend(self.sp * i for i in range(1, self.P + 1))
        lags = sorted(set(lags))

        if len(lags) == 0:
            raise ValueError("NNetAR requires at least one lag input.")

        return np.asarray(lags, dtype=int)

    def _make_training_data(self, y_values):
        """Create lagged feature matrix and one-step-ahead target."""
        max_lag = int(np.max(self.lags_))

        if len(y_values) <= max_lag:
            raise ValueError(
                "NNetAR requires more observations than the largest configured lag."
            )

        X_rows = []
        y_rows = []

        for idx in range(max_lag, len(y_values)):
            lagged = y_values[idx - self.lags_]
            target = y_values[idx]

            if np.isnan(target) or np.isnan(lagged).any():
                continue

            X_rows.append(lagged)
            y_rows.append(target)

        return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float)

    def _recursive_predict(self, n_steps):
        """Predict recursively for ``n_steps`` future time points."""
        history = list(self._fit_values_)
        predictions = []

        for _ in range(n_steps):
            x_pred = np.asarray([history[-lag] for lag in self.lags_], dtype=float)
            x_pred = x_pred.reshape(1, -1)

            if self.scaler_ is not None:
                x_pred = self.scaler_.transform(x_pred)

            model_preds = [model.predict(x_pred)[0] for model in self.models_]
            y_next = float(np.mean(model_preds))
            history.append(y_next)
            predictions.append(y_next)

        return predictions

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {
                "p": 2,
                "P": 0,
                "size": 2,
                "repeats": 1,
                "max_iter": 50,
                "random_state": 0,
            },
            {
                "p": 0,
                "P": 1,
                "sp": 4,
                "size": 2,
                "repeats": 1,
                "max_iter": 50,
                "random_state": 0,
            },
        ]
