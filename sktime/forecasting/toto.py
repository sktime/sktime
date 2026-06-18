# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements ToTo forecaster."""

# This product includes software developed at Datadog, Copyright 2025 Datadog, Inc.

__author__ = [
    "JATAYU000",
    "bthecohen",
    "anna-monica",
    "vendettacoder",
    "clettieri",
    "abdulfatir",
    "EmaadKhwaja",
    "sdavtaker",
    "ViktoriyaZhukova",
    "rostami-dd",
    "chenghaoliu89",
    "dsask",
    "othmaneabou",
    "daniellekutner",
    "DresdenGman",
]
__all__ = ["TotoForecaster"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TotoForecaster(BaseForecaster):
    """Toto foundation model forecaster for zero-shot forecasting.

    Direct interface to forecaster from DataDog/toto [1]_.

    Toto is a foundation model for multivariate time series forecasting with a focus on
    observability metrics. This model leverages innovative architectural designs to
    efficiently handle the high-dimensional, complex time series that are characteristic
    of observability data. Generate both point forecasts and uncertainty estimates using
    a Student-T mixture model. Support for variable prediction horizons and context
    lengths.

    Exogenous variables passed as ``X`` to ``fit`` and ``predict`` are forwarded
    to the underlying Toto model as ``future_exogenous_variables``. The columns
    present in ``X`` at fit time must also be present at predict time.

    Parameters
    ----------
    num_samples : int
        Number of samples for probabilistic forecasting
    samples_per_batch : int, optional (default=1)
        Control memory usage during inference
    prediction_type : string, optional (default='median')
        Type of prediction to generate ('mean' or 'median').
    scale_factor_exponent : int, optional (default=10)
        Exponent for the scale factor used in the model.
    stabilize_with_global : boolean, optional (default=True)
        Whether to stabilize the model with global context.
    use_memory_efficient_attention : boolean, optional (default=True)
        Whether to use memory-efficient attention mechanisms using Xformers.
    model_path : string, optional (default='Datadog/Toto-Open-Base-1.0')
        Path to the Toto huggingface model.
    device : string, optional (default=None)
        Specifies the device on which to run the model on ('cpu' or 'cuda').

    References
    ----------
    .. [1] https://github.com/DataDog/toto

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.toto import TotoForecaster
    >>> _, y = load_longley()
    >>> model = TotoForecaster()
    >>> model.fit(y)
    TotoForecaster()
    >>> forecast = model.predict(fh=[1,2,5])
    """

    _tags = {
        "y_inner_mtype": ["pd.DataFrame"],
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        # contribution and dependency tags
        "authors": [
            "JATAYU000",
            "bthecohen",
            "anna-monica",
            "vendettacoder",
            "clettieri",
            "abdulfatir",
            "EmaadKhwaja",
            "sdavtaker",
            "ViktoriyaZhukova",
            "rostami-dd",
            "chenghaoliu89",
            "dsask",
            "othmaneabou",
            "daniellekutner",
            "DresdenGman",
        ],
        "maintainers": ["JATAYU000"],
        "python_version": ">= 3.10",
        "python_dependencies": ["torch>=2.5", "toto-ts>=0.1.3"],
        # CI and test flags
        # -----------------
        "tests:vm": True,  # run tests on own VM?
    }

    def __init__(
        self,
        seed=None,
        num_samples: int = 1,
        samples_per_batch: int = 1,
        prediction_type: str = "median",
        scale_factor_exponent: int = 10,
        stabilize_with_global: bool = True,
        use_memory_efficient_attention: bool = False,
        model_path: str = "Datadog/Toto-Open-Base-1.0",
        device=None,
    ):
        self.model_path = model_path
        self.device = device
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.use_memory_efficient_attention = use_memory_efficient_attention
        if self.use_memory_efficient_attention:
            if _check_soft_dependencies("xformers", severity="warning"):
                self.set_tags(python_dependencies=["torch", "xformers", "accelerate"])
            else:
                raise ImportError(
                    """xformers is required for memory efficient attention.
                    Refer to https://github.com/facebookresearch/xformers"""
                )
        self.stabilize_with_global = stabilize_with_global
        self.scale_factor_exponent = scale_factor_exponent
        self.prediction_type = prediction_type
        if prediction_type not in ["mean", "median"]:
            raise ValueError("prediction_type must be either 'mean' or 'median'")
        self.seed = seed
        self._seed = np.random.randint(0, 2**31) if seed is None else seed
        super().__init__()

    def _get_toto_key(self):
        """Get a unique key for the Toto model based on configuration parameters."""
        kwargs = self._get_toto_kwargs()
        key = {**kwargs, "device": self._device}
        return str(sorted(key.items()))

    def _get_toto_kwargs(self):
        """Get keyword arguments for the Toto model."""
        return {
            "pretrained_model_name_or_path": self.model_path,
            "use_memory_efficient_attention": self.use_memory_efficient_attention,
            "stabilize_with_global": self.stabilize_with_global,
            "scale_factor_exponent": self.scale_factor_exponent,
        }

    def _X_to_tensor(self, X, device):
        """Convert a pd.DataFrame of exogenous variables to a float32 torch tensor.

        The tensor shape expected by Toto is (n_exog_cols, n_timepoints),
        matching the layout used for the target series.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_timepoints, n_exog_cols), or None
        device : str
            Torch device string, e.g. 'cpu' or 'cuda'.

        Returns
        -------
        torch.Tensor of shape (n_exog_cols, n_timepoints), or None if X is None.
        """
        import torch

        if X is None:
            return None
        return torch.tensor(X.values.T, dtype=torch.float32).to(device)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series to fit.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables. If provided, column names are stored so
            that predict-time X can be validated for consistency.
        fh : ForecastingHorizon, optional (default=None)
            The forecast horizon.

        Returns
        -------
        self : TotoForecaster
        """
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        if self.device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        self.input_series = torch.tensor(
            y.values.T, dtype=torch.float32
        ).to(self._device)
        self._id_mask = torch.zeros_like(self.input_series).to(self._device)
        self._padding_mask = torch.full_like(
            self.input_series, True, dtype=torch.bool
        ).to(self._device)
        self.timestamp_seconds = torch.zeros_like(self.input_series)
        self.time_interval_seconds = torch.full(
            (self.input_series.shape[0],), 60 * 15, dtype=torch.float32
        ).to(self._device)
        self._series = MaskedTimeseries(
            series=self.input_series,
            padding_mask=self._padding_mask,
            id_mask=self._id_mask,
            timestamp_seconds=self.timestamp_seconds,
            time_interval_seconds=self.time_interval_seconds,
        )

        # Store exogenous column names for predict-time validation.
        # The historical X values are not passed to Toto during fit because
        # the model is zero-shot; only future exogenous values matter at
        # predict time.
        self._X_columns = list(X.columns) if X is not None else None

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecast horizon.
        X : pd.DataFrame, optional (default=None)
            Future exogenous variables aligned to ``fh``. If provided at fit
            time, X must also be provided here with the same columns.

        Returns
        -------
        y_pred : pd.DataFrame
        """
        import torch

        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

        prediction_length = max(fh.to_relative(self._cutoff))

        forecaster = _CachedTotoForecaster(
            key=self._get_toto_key(),
            toto_kwargs=self._get_toto_kwargs(),
            device=self._device,
        ).load_from_checkpoint()

        # Convert future exogenous variables to tensor if provided.
        # Shape: (n_exog_cols, prediction_length)
        future_exog_tensor = self._X_to_tensor(X, self._device)

        forecast = forecaster.forecast(
            self._series,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
            future_exogenous_variables=future_exog_tensor,
        )

        if self.prediction_type.lower() == "median":
            all_predictions = forecast.median.cpu().squeeze(0).numpy().T
        else:
            all_predictions = forecast.mean.cpu().squeeze(0).numpy().T

        pred_index = fh.to_absolute(self._cutoff)._values
        relative_indices = fh.to_relative(self._cutoff) - 1
        selected_predictions = all_predictions[relative_indices]

        y_pred = pd.DataFrame(
            selected_predictions, index=pred_index, columns=self._y.columns
        )
        return y_pred

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast.

        Parameters
        ----------
        fh : ForecastingHorizon
        X : pd.DataFrame, optional
            Future exogenous variables aligned to ``fh``.
        alpha : list[float]
            Quantile levels to compute.

        Returns
        -------
        pred_quantiles : pd.DataFrame
        """
        import torch

        prediction_length = max(fh.to_relative(self._cutoff))

        forecaster = _CachedTotoForecaster(
            key=self._get_toto_key(),
            toto_kwargs=self._get_toto_kwargs(),
            device=self._device,
        ).load_from_checkpoint()

        # Convert future exogenous variables to tensor if provided.
        future_exog_tensor = self._X_to_tensor(X, self._device)

        forecast = forecaster.forecast(
            self._series,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
            future_exogenous_variables=future_exog_tensor,
        )

        var_names = self._y.columns
        cols_idx = pd.MultiIndex.from_product([var_names, alpha])
        pred_index = fh.to_absolute(self._cutoff)._values
        relative_indices = fh.to_relative(self._cutoff) - 1
        pred_quantiles = pd.DataFrame(index=pred_index, columns=cols_idx)
        alpha_tensor = torch.tensor(alpha, device=self._device)
        quantiles = forecast.quantile(alpha_tensor)
        if quantiles.dim() > 3:
            quantile_values = quantiles.cpu().squeeze(1).numpy()
        else:
            quantile_values = quantiles.cpu().numpy()
        for i, var_name in enumerate(var_names):
            for j, a in enumerate(alpha):
                selected_quantiles = quantile_values[j, i, relative_indices]
                pred_quantiles[(var_name, a)] = selected_quantiles
        return pred_quantiles

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        test_params = [
            {"num_samples": 2, "samples_per_batch": 2, "prediction_type": "median"},
            {"num_samples": 2, "samples_per_batch": 1, "prediction_type": "mean"},
            {"num_samples": 1, "samples_per_batch": 1, "prediction_type": "mean"},
        ]
        return test_params


@_multiton
class _CachedTotoForecaster:
    """Cached Toto forecaster."""

    def __init__(self, key, toto_kwargs, device):
        self.key = key
        self.toto_kwargs = toto_kwargs
        self.device = device
        self.forecaster = None

    def load_from_checkpoint(self):
        if self.forecaster is not None:
            return self.forecaster
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        toto_model = Toto.from_pretrained(**self.toto_kwargs)
        toto_model.to(self.device)
        toto_model.compile()
        self.forecaster = TotoForecaster(toto_model.model)
        return self.forecaster
