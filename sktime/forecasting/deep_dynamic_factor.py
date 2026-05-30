# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Deep Dynamic Factor Model (DDFM) forecaster.

Implements a deep dynamic factor model using PyTorch. The model learns latent
factors via an autoencoder trained with a denoising loop, then forecasts via
state-space factor dynamics and linear decoding.
"""

from __future__ import annotations

__author__ = ["minkeymouse"]
__all__ = ["DeepDynamicFactor"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.representation_learning._autoencoder import (
    _SimpleAutoencoder,
    _extract_decoder_params,
)
from sktime.forecasting.representation_learning._dataset import (
    _DDFMDataset,
)
from sktime.forecasting.representation_learning._state_space import (
    _FittedFactorModel,
    _generic_var1_dynamics,
    _linear_decoder_ddfm_dynamics,
    _StateSpaceParams,
)
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.dynamic_factors import (
    estimate_var,
    forecast_factors_ar1,
    get_idio,
)


class _DDFMInternal:
    """Core fit, update, and predict logic for the DeepDynamicFactor forecaster."""

    def __init__(
        self,
        *,
        encoder_size: tuple[int, ...],
        decoder_type: str,
        max_iter: int,
        n_mc_samples: int,
        window_size: int,
        learning_rate: float,
        tolerance: float,
        random_state: int | None,
    ):
        self.encoder_size = encoder_size
        self.decoder_type = decoder_type
        self.max_iter = int(max_iter)
        self.n_mc_samples = int(n_mc_samples)
        self.window_size = int(window_size)
        self.learning_rate = float(learning_rate)
        self.tolerance = float(tolerance)
        self.random_state = int(random_state) if random_state is not None else None

        self._fitted: _FittedFactorModel | None = None
        self._parts = None
        self._data = None

    @property
    def fitted_(self) -> _FittedFactorModel:
        if self._fitted is None:
            raise RuntimeError("_DDFMInternal is not fitted")
        return self._fitted

    def fit(self, y, X=None, *, scaler):
        _check_soft_dependencies("torch", severity="error")
        import random
        import torch

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pd.DataFrame")
        if X is not None and not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pd.DataFrame or None")

        # Column names from sktime scenarios can overlap between X and y (e.g. both
        # use integer column labels). We therefore re-label internally to ensure
        # a clean covariate/target split.
        if X is not None:
            X_ = X.copy()
            y_ = y.copy()
            X_.columns = [f"X__{i}" for i in range(X_.shape[1])]
            y_.columns = [f"y__{i}" for i in range(y_.shape[1])]
            data_df = pd.concat([X_, y_], axis=1)
            covariates = list(X_.columns)
            self._covariate_columns_ = list(X_.columns)
            self._target_columns_ = list(y_.columns)
        else:
            y_ = y.copy()
            y_.columns = [f"y__{i}" for i in range(y_.shape[1])]
            data_df = y_
            covariates = []
            self._covariate_columns_ = []
            self._target_columns_ = list(y_.columns)

        dataset = _DDFMDataset(data=data_df, covariates=covariates, scaler=scaler)
        self._data = dataset

        rng = (
            np.random.RandomState(self.random_state)
            if self.random_state is not None
            else np.random.RandomState()
        )

        data_imputed = pd.DataFrame(
            np.asarray(dataset.data.values, dtype=np.float64).copy(),
            index=dataset.data.index,
            columns=dataset.data.columns,
        )
        data_imputed = (
            data_imputed.interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
        data_denoised = data_imputed.copy()

        device = torch.device("cpu")
        torch_dtype = torch.float32

        # tensors are created inside dataset factory methods; keep local scope minimal

        pretrain_ds = dataset.create_pretrain_dataset(data_imputed, device=device)

        decoder_size = None
        if self.decoder_type == "mlp" and len(self.encoder_size) > 1:
            decoder_size = tuple(reversed(self.encoder_size[:-1]))

        autoencoder = _SimpleAutoencoder.from_dataset(
            pretrain_ds,
            encoder_size=self.encoder_size,
            decoder_size=decoder_size,
            decoder_type=self.decoder_type,
            activation="relu",
            seed=self.random_state,
        )

        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(self.n_mc_samples), gamma=0.96
        )

        y_actual, factors_mean, eps = self._run_denoising_loop(
            dataset=dataset,
            autoencoder=autoencoder,
            data_imputed=data_imputed,
            data_denoised=data_denoised,
            optimizer=optimizer,
            scheduler=scheduler,
            rng=rng,
            device=device,
        )

        state = self._build_state_space_from_training(
            dataset=dataset,
            autoencoder=autoencoder,
            factors_mean=factors_mean,
            eps=eps,
            y_actual=y_actual,
        )

        self._parts = autoencoder
        self._fitted = _FittedFactorModel(factors=factors_mean, state_space=state)
        return self

    def _run_denoising_loop(
        self,
        *,
        dataset: _DDFMDataset,
        autoencoder: _SimpleAutoencoder,
        data_imputed: pd.DataFrame,
        data_denoised: pd.DataFrame,
        optimizer,
        scheduler,
        rng: np.random.RandomState,
        device,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the DDFM-style denoising loop and return targets, factors, residuals.

        Parameters
        ----------
        dataset : _DDFMDataset
            Target/covariate container and batch factories.
        autoencoder : _SimpleAutoencoder
            Fitted autoencoder (pretrained once inside, then updated each iteration).
        data_imputed : pd.DataFrame
            Imputed panel; may be updated in-place for missing targets.
        data_denoised : pd.DataFrame
            Denoised panel; updated each iteration.
        optimizer : torch.optim.Optimizer
            Optimizer for autoencoder training.
        scheduler : torch.optim.lr_scheduler.LRScheduler
            Learning-rate scheduler.
        rng : np.random.RandomState
            RNG for MC noise draws.
        device : torch.device
            Device for tensors.

        Returns
        -------
        y_actual : np.ndarray
            Scaled target values (n_timepoints, n_targets).
        factors_mean : np.ndarray
            Mean latent factors over MC samples (n_timepoints, n_factors).
        eps : np.ndarray
            Residuals after final iteration (n_timepoints, n_targets).
        """
        import torch

        pretrain_ds = dataset.create_pretrain_dataset(data_imputed, device=device)

        use_mse_loss = True
        try:
            y_clean_np = pretrain_ds.y_clean.detach().cpu().numpy()
            use_mse_loss = (not np.isnan(y_clean_np).any()) and (len(pretrain_ds) >= 50)
        except Exception:
            use_mse_loss = len(pretrain_ds) >= 50

        autoencoder.pretrain(
            pretrain_ds.full_input,
            pretrain_ds.y_clean,
            epochs=int(self.n_mc_samples),
            batch_size=max(1, min(self.window_size, len(pretrain_ds))),
            optimizer=optimizer,
            use_mse_loss=use_mse_loss,
        )

        with torch.no_grad():
            y_pred0 = (
                autoencoder.predict(pretrain_ds.full_input)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        y_actual = dataset.y.astype(np.float64, copy=False)
        eps = y_actual - y_pred0
        y_pred_prev = None
        factors_mean = None

        for _iter in range(self.max_iter):
            Phi, mu_eps, std_eps = get_idio(eps, dataset.observed_y, min_obs=5)

            if eps.shape[0] >= 2:
                eps_denoise = eps[:-1, :] @ Phi
                denoised_arr = np.asarray(data_denoised.values, dtype=np.float64).copy()
                denoised_arr[1:, dataset.target_indices] = (
                    data_imputed.values[1:, dataset.target_indices].astype(np.float64)
                    - eps_denoise
                )
                data_denoised = pd.DataFrame(
                    denoised_arr, index=data_denoised.index, columns=data_denoised.columns
                )
            data_denoised = (
                data_denoised.interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
            )

            X_tmp, y_tmp = dataset.split_features_and_targets(data_denoised)
            X_tmp = X_tmp if X_tmp is not None else np.empty((len(y_tmp), 0))

            autoencoder_datasets = dataset.create_autoencoder_datasets_list(
                n_mc_samples=self.n_mc_samples,
                mu_eps=mu_eps,
                std_eps=std_eps,
                X=X_tmp,
                y_tmp=y_tmp.values,
                y_actual=y_actual,
                rng=rng,
                device=device,
            )

            for ae_ds in autoencoder_datasets:
                autoencoder.fit(
                    dataset=ae_ds,
                    epochs=1,
                    batch_size=max(1, min(self.window_size, len(ae_ds))),
                    learning_rate=self.learning_rate,
                    optimizer_type="Adam",
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

            with torch.no_grad():
                factors_list = [
                    autoencoder.encoder(ae_ds.full_input) for ae_ds in autoencoder_datasets
                ]
                factors_tensor = torch.stack(factors_list, dim=0)
                y_pred_samples = torch.stack(
                    [autoencoder.decoder(f) for f in factors_list], dim=0
                )

                y_pred_full = (
                    torch.mean(y_pred_samples, dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                factors_mean = (
                    torch.mean(factors_tensor, dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

            y_pred = y_pred_full

            missing_y = dataset.missing_y
            if missing_y.any():
                data_imputed.values[:, dataset.target_indices][missing_y] = y_pred[missing_y]
            eps = data_imputed.values[:, dataset.target_indices] - y_pred

            if y_pred_prev is not None:
                denom = max(1e-12, float(np.linalg.norm(y_pred_prev)))
                if float(np.linalg.norm(y_pred - y_pred_prev)) / denom < self.tolerance:
                    break
            y_pred_prev = y_pred.copy()

        if factors_mean is None:
            raise RuntimeError("Denoising loop did not produce any factors")

        return y_actual, factors_mean, eps

    def _build_state_space_from_training(
        self,
        *,
        dataset: _DDFMDataset,
        autoencoder: _SimpleAutoencoder,
        factors_mean: np.ndarray,
        eps: np.ndarray,
        y_actual: np.ndarray,
    ) -> _StateSpaceParams:
        """Build state-space (F, H, b) from fitted factors and decoder/observations."""
        if self.decoder_type == "linear":
            decoder_weight, _bias = _extract_decoder_params(autoencoder.decoder)
            return _linear_decoder_ddfm_dynamics(
                factors=factors_mean,
                eps=eps,
                decoder_weight=decoder_weight,
                observed_y=dataset.observed_y,
            )
        return _generic_var1_dynamics(
            factors=factors_mean,
            y_scaled=y_actual,
        )

    def update(self, y, X=None, *, column_order=None):
        _check_soft_dependencies("torch", severity="error")
        import torch

        if self._data is None or self._parts is None:
            raise RuntimeError("Call fit before update")

        # Recreate the internal [X|y] schema used during fit.
        cov_cols = getattr(self, "_covariate_columns_", [])
        tgt_cols = getattr(self, "_target_columns_", [])

        y_ = y.copy()
        if tgt_cols:
            if y_.shape[1] != len(tgt_cols):
                raise ValueError(
                    "y in update must have same number of columns as in fit "
                    f"({len(tgt_cols)}), got {y_.shape[1]}"
                )
            y_.columns = list(tgt_cols)
        else:
            y_.columns = [f"y__{i}" for i in range(y_.shape[1])]

        if cov_cols:
            if X is None:
                X_ = pd.DataFrame(np.nan, index=y_.index, columns=list(cov_cols))
            else:
                X_ = X.copy()
                if X_.shape[1] != len(cov_cols):
                    raise ValueError(
                        "X in update must have same number of columns as in fit "
                        f"({len(cov_cols)}), got {X_.shape[1]}"
                    )
                X_.columns = list(cov_cols)
            data_df = pd.concat([X_, y_], axis=1)
        else:
            data_df = y_
        new_ds = _DDFMDataset.from_dataset(data_df, self._data)
        device = torch.device("cpu")
        torch_dtype = torch.float32

        x_df = new_ds.data
        x_df = x_df.interpolate(method="linear", limit_direction="both").ffill().bfill()
        X_tmp, y_tmp = new_ds.split_features_and_targets(x_df)
        if X_tmp is None:
            full_input = torch.from_numpy(y_tmp.values).to(dtype=torch_dtype, device=device)
        else:
            full_input = torch.from_numpy(
                np.concatenate([X_tmp.values, y_tmp.values], axis=1)
            ).to(dtype=torch_dtype, device=device)

        enc = self._parts.encoder
        enc.eval()
        with torch.no_grad():
            z_new = enc(full_input).detach().cpu().numpy().astype(np.float64)

        fitted = self.fitted_
        fitted.factors = np.concatenate([fitted.factors, z_new], axis=0)
        fitted.state_space.F, _ = estimate_var(fitted.factors, order=1)
        return self

    def predict(self, *, horizon: int, return_factors: bool = False):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        fitted = self.fitted_
        dataset = self._data
        if dataset is None:
            raise RuntimeError("No fitted data found")

        z_last = fitted.factors[-1]
        z_fore = forecast_factors_ar1(z_last, fitted.state_space.F, horizon=horizon)

        y_scaled = z_fore @ fitted.state_space.H.T + fitted.state_space.b
        if dataset.scaler is not None:
            y_pred = dataset.scaler.inverse_transform(y_scaled)
        else:
            y_pred = y_scaled
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        if return_factors:
            return y_pred, z_fore
        return y_pred, None


class DeepDynamicFactor(BaseForecaster):
    """Deep Dynamic Factor Model forecaster.

    Uses a neural autoencoder to learn latent factors, a denoising training
    loop, and a state-space forecasting layer that rolls factors forward
    and decodes them to targets.

    Notes
    -----
    - Exogenous ``X`` is used only in ``fit``/``update`` as additional encoder
      inputs (to help learn factors). Forecasts are produced purely from the
      learned factor/state-space dynamics, so passing ``X`` to ``predict`` has
      no effect.
    - ``update(y, X, update_params=True)`` runs the fitted encoder on new
      ``[X|y]`` (if provided) to append factors and re-estimate the transition
      matrix; ``update_params=False`` only moves the cutoff.

    Parameters
    ----------
    encoder_size : tuple of int, optional (default=(16, 4))
        Encoder layer sizes. Last element is the number of factors.
    decoder_type : str, optional (default='mlp')
        Decoder type: 'linear' or 'mlp'.
    max_iter : int, optional (default=50)
        Maximum MCMC denoising iterations.
    n_mc_samples : int, optional (default=10)
        Number of MC samples per MCMC iteration.
    window_size : int, optional (default=10)
        Training window size (batch size for autoencoder updates).
    learning_rate : float, optional (default=0.005)
        Learning rate for Adam optimizer in the autoencoder training loop.
    tolerance : float, optional (default=0.0005)
        Convergence tolerance for MCMC loop.
    random_state : int, optional (default=42)
        Random state for reproducibility.

    Attributes
    ----------
    _forecaster_ : fitted estimator
        Wrapped model instance after fit.
    _column_order_ : list of str
        Column order of fit data; used for update to preserve schema.

    References
    ----------
    .. [1] Deep dynamic factor models: a class of deep learning models for
       dynamic factor analysis (see also dfm-python). Related: state-space
       formulations where common factors and idiosyncratic components are
       unobserved states; nowcasting and conditional latent representations.
    """

    _tags = {
        "authors": ["minkeymouse"],
        "maintainers": ["minkeymouse"],
        "python_dependencies": ["torch", "scikit-learn"],
        "scitype:y": "both",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        # accepts X in fit/update (used to learn factors); predict accepts X but ignores it
        "capability:exogenous": True,
        "capability:missing_values": True,
        "capability:random_state": True,
        "property:randomness": "derandomized",
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "capability:insample": False,
        "capability:pred_int": False,
    }

    def __init__(
        self,
        encoder_size=(16, 4),
        decoder_type="mlp",
        max_iter=50,
        n_mc_samples=10,
        window_size=10,
        learning_rate=0.005,
        tolerance=0.0005,
        random_state=42,
    ):
        if decoder_type not in ("linear", "mlp"):
            raise ValueError(
                f"decoder_type must be 'linear' or 'mlp', got {decoder_type!r}"
            )
        encoder_size = tuple(encoder_size)
        if len(encoder_size) < 1 or encoder_size[-1] < 1:
            raise ValueError(
                "encoder_size must be a non-empty sequence with last element >= 1 "
                f"(number of factors), got {encoder_size}"
            )
        if max_iter < 1 or n_mc_samples < 1 or window_size < 1:
            raise ValueError("max_iter, n_mc_samples, and window_size must be >= 1")
        self.encoder_size = encoder_size
        self.decoder_type = decoder_type
        self.max_iter = max_iter
        self.n_mc_samples = n_mc_samples
        self.window_size = window_size
        self.learning_rate = float(learning_rate)
        self.tolerance = float(tolerance)
        self.random_state = int(random_state) if random_state is not None else None
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series (multivariate).
        X : pd.DataFrame, optional
            Exogenous variables (covariates).
        fh : ForecastingHorizon, optional
            Not used.

        Returns
        -------
        self
        """
        from sklearn.preprocessing import StandardScaler

        if len(y) < self.window_size:
            raise ValueError(
                f"y has {len(y)} observations but window_size is "
                f"{self.window_size}; need len(y) >= window_size"
            )

        if X is not None:
            data = pd.concat([X, y], axis=1)
        else:
            data = y.copy()

        model = _DDFMInternal(
            encoder_size=self.encoder_size,
            decoder_type=self.decoder_type,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_mc_samples=self.n_mc_samples,
            window_size=self.window_size,
            learning_rate=self.learning_rate,
            tolerance=self.tolerance,
        )
        scaler = StandardScaler()
        model.fit(y=y, X=X, scaler=scaler)
        self._forecaster_ = model  # sktime convention: trailing underscore for fitted
        self._column_order_ = list(data.columns)
        return self

    def _update(self, y, X=None, update_params=True):
        """Update cutoff and optionally update the fitted forecaster with new data.

        Parameters
        ----------
        y : pd.DataFrame
            New target observations.
        X : pd.DataFrame, optional
            New exogenous variables.
        update_params : bool
            If True, extend factors via the fitted model's update.

        Returns
        -------
        self
        """
        if not update_params:
            return self
        self._forecaster_.update(y=y, X=X, column_order=self._column_order_)
        return self

    def _fh_to_horizon_and_step_selector(self, fh):
        """Resolve fh to prediction index, horizon length, and step selector.

        Returns
        -------
        pred_index : pd.Index
        horizon : int
        step_selector : np.ndarray or slice
            Indexer into model output (0-based); apply as y_pred[step_selector].
        """
        fh_rel = fh.to_relative(self.cutoff)
        try:
            fh_steps = np.asarray(fh_rel.to_pandas(), dtype=np.float64)
        except Exception:
            fh_steps = np.asarray(fh_rel, dtype=np.float64)
        pred_index = fh.to_absolute_index(self.cutoff)
        n_steps = len(pred_index)
        if n_steps == 0:
            return pred_index, 1, slice(0, 0)
        if fh_steps.size > 0:
            # sktime horizons are discrete; reject non-integer steps instead of rounding
            if not np.all(np.isfinite(fh_steps)):
                raise ValueError("fh contains non-finite steps")
            if not np.allclose(fh_steps, np.round(fh_steps)):
                raise ValueError("fh must contain integer steps only")
            step_one_based = fh_steps.astype(np.intp)
            if np.any(step_one_based < 1):
                raise ValueError("fh must contain only positive steps (>= 1)")
            horizon = int(np.max(step_one_based))
            step_selector = step_one_based - 1
        else:
            horizon = 1
            step_selector = slice(None, n_steps)
        return pred_index, horizon, step_selector

    def _predict(self, fh, X=None):
        """Forecast using the fitted state-space model.

        Parameters
        ----------
        fh : ForecastingHorizon
            Horizon (relative to cutoff).
        X : pd.DataFrame, optional
            Exogenous data; unused (forecasts from state-space only).

        Returns
        -------
        pd.DataFrame
            Point forecast; index = fh.to_absolute_index(cutoff), columns = targets.
        """
        pred_index, horizon, step_selector = self._fh_to_horizon_and_step_selector(fh)
        if len(pred_index) == 0:
            return pd.DataFrame(columns=self._y.columns)
        y_pred, _ = self._forecaster_.predict(horizon=horizon, return_factors=True)
        y_pred = y_pred[step_selector]
        return pd.DataFrame(y_pred, index=pred_index, columns=self._y.columns)

    def _get_fitted_params(self):
        """Return fitted parameters (factors, model, state_space_F/H when available)."""
        forecaster = self._forecaster_
        fitted = getattr(forecaster, "_fitted", None)
        out = {
            "factors": getattr(fitted, "factors", None) if fitted is not None else None,
            "model": forecaster,
        }
        if fitted is not None and getattr(fitted, "state_space", None) is not None:
            ss = fitted.state_space
            out["state_space_F"] = ss.F
            out["state_space_H"] = ss.H
        return out

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameter sets for the forecaster.

        Fast settings (few iterations, small encoder) for unit tests.
        """
        if parameter_set != "default":
            return []
        base = {
            "encoder_size": (8, 2),
            "max_iter": 2,
            "n_mc_samples": 2,
            "window_size": 4,
            "random_state": 42,
        }
        return [
            base,
            {**base, "encoder_size": (6, 2), "decoder_type": "mlp"},
        ]


class _RepresentationLearningForecasterPlaceholder(BaseForecaster):
    """Reserved for a future representation-learning forecaster; not implemented."""

    _tags = {
        "authors": ["minkeymouse"],
        "maintainers": ["minkeymouse"],
        "scitype:y": "both",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": True,
        "capability:missing_values": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        raise NotImplementedError(
            "_RepresentationLearningForecasterPlaceholder is a non-public "
            "placeholder and must not be used directly."
        )

    def _predict(self, fh, X=None):
        raise NotImplementedError(
            "_RepresentationLearningForecasterPlaceholder is a non-public "
            "placeholder and must not be used directly."
        )
