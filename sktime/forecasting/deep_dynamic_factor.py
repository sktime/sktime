# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Deep Dynamic Factor Model (DDFM) forecaster.

Wraps dfm-python's DDFM: factor model with autoencoder and MCMC-style denoising
training, then state-space forecasting. Requires torch and dfm_python.
"""

__author__ = ["minkeymouse"]
__all__ = ["DeepDynamicFactor"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster

# Sentinel for time index: not a column name, so DDFMDataset uses data.index.
_SKTIME_TIME_IDX = "__sktime_tidx__"


class DeepDynamicFactor(BaseForecaster):
    """Deep Dynamic Factor Model forecaster.

    Interface to the Deep Dynamic Factor Model (DDFM) from the dfm-python package.
    DDFM uses a neural autoencoder to extract latent factors, MCMC-style denoising
    training, and a state-space layer for forecasting. Unlike the linear
    statsmodels DynamicFactor, DDFM supports nonlinear factor extraction and
    optional exogenous covariates (used for encoding, not forecasted).

    Parameters
    ----------
    encoder_size : tuple of int, optional (default=(16, 4))
        Encoder layer sizes. Last element is the number of factors.
    decoder_type : str, optional (default='linear')
        Decoder type: 'linear' or 'mlp'.
    max_iter : int, optional (default=50)
        Maximum MCMC denoising iterations.
    n_mc_samples : int, optional (default=10)
        Number of MC samples per MCMC iteration.
    window_size : int, optional (default=10)
        Training window size (batch size for autoencoder updates).
    tolerance : float, optional (default=0.0005)
        Convergence tolerance for MCMC loop.
    seed : int, optional (default=42)
        Random seed for reproducibility.

    Attributes
    ----------
    _ddfm_ : DDFM
        Fitted dfm-python DDFM instance (after fit).
    _target_columns_ : list of str
        Names of target columns from y; used as prediction output columns.

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
        "python_dependencies": ["torch", "dfm_python"],
        "scitype:y": "multivariate",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": True,
        "capability:missing_values": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "capability:insample": False,
        "capability:pred_int": False,
    }

    def __init__(
        self,
        encoder_size=(16, 4),
        decoder_type="linear",
        max_iter=50,
        n_mc_samples=10,
        window_size=10,
        tolerance=0.0005,
        seed=42,
    ):
        self.encoder_size = encoder_size
        self.decoder_type = decoder_type
        self.max_iter = max_iter
        self.n_mc_samples = n_mc_samples
        self.window_size = window_size
        self.tolerance = tolerance
        self.seed = seed
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit DDFM to training data.

        Builds DDFMDataset from y (and X as covariates), fits the DDFM,
        and builds the state-space model for forecasting.
        """
        from sklearn.preprocessing import StandardScaler

        from dfm_python.dataset.ddfm_dataset import DDFMDataset
        from dfm_python.models.ddfm.ddfm import DDFM

        # y is DataFrame by base contract (y_inner_mtype="pd.DataFrame")
        self._target_columns_ = list(y.columns)

        # Build data: [X | y] so DDFM sees covariates then targets
        if X is not None:
            data = pd.concat([X, y], axis=1)
            covariates = list(X.columns)
        else:
            data = y.copy()
            covariates = None

        scaler = StandardScaler()
        dataset = DDFMDataset(
            data=data,
            time_idx=_SKTIME_TIME_IDX,
            covariates=covariates,
            scaler=scaler,
        )
        ddfm = DDFM(
            dataset=dataset,
            encoder_size=tuple(self.encoder_size),
            decoder_type=self.decoder_type,
            seed=self.seed,
            max_iter=self.max_iter,
            n_mc_samples=self.n_mc_samples,
            window_size=self.window_size,
            tolerance=self.tolerance,
        )
        ddfm.fit()
        ddfm.build_state_space()
        self._ddfm_ = ddfm
        return self

    def _predict(self, fh, X=None):
        """Forecast using the fitted state-space model.

        Parameters
        ----------
        fh : ForecastingHorizon
            Horizon (relative to cutoff).
        X : pd.DataFrame, optional
            Exogenous data; unused (DDFM forecasts from state-space only).

        Returns
        -------
        pd.DataFrame
            Point forecast, index = fh.to_absolute_index(cutoff), columns = target columns.
        """
        fh_rel = fh.to_relative(self.cutoff)
        try:
            fh_steps = np.asarray(fh_rel.to_pandas())
        except Exception:
            fh_steps = np.asarray(fh_rel)
        horizon = int(np.max(fh_steps)) if fh_steps.size > 0 else 1
        horizon = max(1, horizon)
        y_pred, _ = self._ddfm_.predict(
            horizon=horizon, return_series=True, return_factors=True
        )
        pred_index = fh.to_absolute_index(self.cutoff)
        n_steps = len(pred_index)
        if n_steps == 0:
            return pd.DataFrame(columns=self._target_columns_)
        # DDFM returns steps 1..horizon; map fh steps to 0-based indices.
        if fh_steps.size > 0 and np.issubdtype(fh_steps.dtype, np.integer):
            step_indices = np.asarray(fh_steps, dtype=np.intp) - 1
            step_indices = np.clip(step_indices, 0, horizon - 1)
            y_pred = y_pred[step_indices]
        else:
            y_pred = y_pred[:n_steps]
        return pd.DataFrame(
            y_pred, index=pred_index, columns=self._target_columns_
        )

    def _get_fitted_params(self):
        """Return fitted parameters including factors and state-space matrices.

        Returns
        -------
        dict
            Keys: "factors" (T x n_factors), "ddfm" (fitted DDFM),
            "state_space_F", "state_space_H" when available.
        """
        ddfm = self._ddfm_
        factors = getattr(ddfm, "factors", None)
        if factors is not None and hasattr(ddfm, "_get_averaged_factors"):
            factors = ddfm._get_averaged_factors()
        out = {"factors": factors, "ddfm": ddfm}
        training_state = getattr(ddfm, "training_state", None)
        if training_state is not None:
            out["state_space_F"] = getattr(training_state, "F", None)
            out["state_space_H"] = getattr(training_state, "H", None)
        return out

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameter sets for the forecaster.

        Fast settings (few iterations, small encoder) for unit tests.
        """
        if parameter_set == "default":
            return [
                {
                    "encoder_size": (8, 2),
                    "max_iter": 2,
                    "n_mc_samples": 2,
                    "window_size": 4,
                    "seed": 42,
                }
            ]
        return []
