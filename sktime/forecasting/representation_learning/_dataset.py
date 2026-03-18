# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dataset and data containers for factor- and autoencoder-based representation learning.

Used by forecasters that learn latent factors or embeddings from time series
(e.g. DeepDynamicFactor). Provides target/covariate split, scaling, and
autoencoder batch factories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sktime.utils.dependencies import _check_soft_dependencies


class _AutoencoderDataset:
    """Container for autoencoder inputs and targets (covariates plus corrupted/clean targets)."""

    def __init__(self, X, y_corrupted, y_clean):
        _check_soft_dependencies("torch", severity="error")
        import torch

        self.X = X
        self.y_corrupted = y_corrupted
        self.y_clean = y_clean
        if self.X is not None:
            self._full_input = torch.cat([self.X, self.y_corrupted], dim=1)
        else:
            self._full_input = self.y_corrupted

    @property
    def full_input(self):
        return self._full_input

    def __len__(self):
        return int(self.y_corrupted.shape[0])


class _DDFMDataset:
    """Dataset for DeepDynamicFactor: target/covariate split, scaling, and autoencoder batch factories."""

    def __init__(self, data: pd.DataFrame, covariates: list[str] | None, scaler):
        _check_soft_dependencies("torch", severity="error")
        import torch

        data = data.copy()
        data.sort_index(inplace=True)

        covariates = [c for c in (covariates or []) if c in data.columns]
        self.covariates = covariates
        self.target_series = [c for c in data.columns if c not in covariates]

        y = data[self.target_series]
        X = data.drop(columns=self.target_series) if self.target_series else pd.DataFrame()

        missing_y = y.isna().values
        observed_y = ~missing_y

        if y.isna().any().any():
            y = y.interpolate(method="linear", limit_direction="both").ffill().bfill()
        if not X.empty and X.isna().any().any():
            X = X.interpolate(method="linear", limit_direction="both").ffill().bfill()

        if scaler is not None and not X.empty:
            X = pd.DataFrame(
                type(scaler)().fit_transform(X.values), index=X.index, columns=X.columns
            )

        if scaler is not None:
            y = pd.DataFrame(
                scaler.fit_transform(y.values), index=y.index, columns=y.columns
            )
        self.scaler = scaler

        self.data = pd.concat([X, y], axis=1) if not X.empty else y
        self.X = X.values if not X.empty else np.empty((len(y), 0))
        self.y = y.values
        self.missing_y = missing_y
        self.observed_y = observed_y

        self._torch_dtype = torch.float32
        self._device = torch.device("cpu")

    @property
    def target_indices(self) -> np.ndarray:
        return np.array([self.data.columns.get_loc(col) for col in self.target_series])

    @property
    def all_columns_are_targets(self):
        return len(self.covariates) == 0

    def split_features_and_targets(self, data: pd.DataFrame):
        if self.all_columns_are_targets:
            return None, data
        X = data.drop(columns=self.target_series)
        y = data[self.target_series]
        return X, y

    @classmethod
    def from_dataset(cls, new_data: pd.DataFrame, dataset: "_DDFMDataset") -> "_DDFMDataset":
        return cls(
            data=new_data,
            covariates=list(dataset.covariates),
            scaler=dataset.scaler,
        )

    def create_autoencoder_dataset(self, X, y_tmp, y_actual, eps_draw):
        return _AutoencoderDataset(X=X, y_corrupted=y_tmp - eps_draw, y_clean=y_actual)

    def create_pretrain_dataset(self, data: pd.DataFrame, device=None):
        _check_soft_dependencies("torch", severity="error")
        import torch

        if device is None:
            device = self._device
        X_df, y_df = self.split_features_and_targets(data)
        y_arr = np.asarray(y_df.values, dtype=np.float64).copy()
        X = (
            None
            if X_df is None
            else torch.from_numpy(np.asarray(X_df.values, dtype=np.float64).copy()).to(
                dtype=self._torch_dtype, device=device
            )
        )
        y = torch.from_numpy(y_arr).to(dtype=self._torch_dtype, device=device)
        return _AutoencoderDataset(X=X, y_corrupted=y, y_clean=y)

    def create_autoencoder_datasets_list(
        self,
        *,
        n_mc_samples: int,
        mu_eps: np.ndarray,
        std_eps: np.ndarray,
        X,
        y_tmp,
        y_actual: np.ndarray,
        rng: np.random.RandomState,
        device=None,
    ):
        _check_soft_dependencies("torch", severity="error")
        import torch

        if device is None:
            device = self._device
        X_array = np.asarray(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=np.float64
        ).copy()
        y_tmp_array = np.asarray(
            y_tmp.values if isinstance(y_tmp, pd.DataFrame) else y_tmp,
            dtype=np.float64,
        ).copy()
        y_actual_arr = np.asarray(y_actual, dtype=np.float64).copy()
        T = y_tmp_array.shape[0]
        eps_draws = rng.multivariate_normal(mu_eps, np.diag(std_eps), (n_mc_samples, T))

        X_tensor = (
            torch.from_numpy(X_array).to(dtype=self._torch_dtype, device=device)
            if X_array.size > 0
            else None
        )
        y_tmp_tensor = torch.from_numpy(y_tmp_array).to(dtype=self._torch_dtype, device=device)
        y_actual_tensor = torch.from_numpy(y_actual_arr).to(
            dtype=self._torch_dtype, device=device
        )
        eps_draws_tensor = torch.from_numpy(eps_draws.copy()).to(
            dtype=self._torch_dtype, device=device
        )

        return [
            self.create_autoencoder_dataset(
                X_tensor, y_tmp_tensor, y_actual_tensor, eps_draws_tensor[i]
            )
            for i in range(n_mc_samples)
        ]
