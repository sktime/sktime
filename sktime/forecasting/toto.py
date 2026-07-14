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
]
__all__ = ["TotoForecaster"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.foundation._base2 import BaseFoundationForecaster


class TotoForecaster(BaseFoundationForecaster):
    """Toto foundation model forecaster for zero-shot forecasting.

    Direct interface to forecaster from DataDog/toto [1]_.

    Toto is a foundation model for multivariate time series forecasting with a focus on
    observability metrics. This model leverages innovative architectural designs to
    efficiently handle the high-dimensional, complex time series that are characteristic
    of observability data. Generate both point forecasts and uncertainty estimates using
    a Student-T mixture model. Support for variable prediction horizons and context
    lengths.

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
        "X_inner_mtype": "None",
        "capability:multivariate": True,
        "capability:exogenous": False,
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
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.use_memory_efficient_attention = use_memory_efficient_attention
        if self.use_memory_efficient_attention:
            if _check_soft_dependencies("xformers", severity="warning"):
                self.set_tags(python_dependencies=["torch", "xformers", "accelerate"])
            else:
                raise ImportError(
                    """
                    xformers is required for memory efficient attention.
                    Refer to https://github.com/facebookresearch/xformers
                    """
                )
        self.stabilize_with_global = stabilize_with_global
        self.scale_factor_exponent = scale_factor_exponent
        self.prediction_type = prediction_type
        if prediction_type not in ["mean", "median"]:
            raise ValueError("prediction_type must be either 'mean' or 'median'")

        self.seed = seed
        self._seed = np.random.randint(0, 2**31) if seed is None else seed
        super().__init__(
            model_path=model_path,
            device=device,
            load_kwargs={
                "pretrained_model_name_or_path": model_path,
                "use_memory_efficient_attention": use_memory_efficient_attention,
                "stabilize_with_global": stabilize_with_global,
                "scale_factor_exponent": scale_factor_exponent,
            },
            random_state=self._seed,
        )

    def _prepare_foundation_context(self, y, X, fh):
        """Fit Toto-specific context state."""
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        if self.device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device
        self.input_series = torch.tensor(y.values.T, dtype=torch.float32).to(
            self._device
        )

        self._id_mask = torch.zeros_like(self.input_series).to(self._device)
        self._padding_mask = torch.full_like(
            self.input_series, True, dtype=torch.bool
        ).to(self._device)

        # current model does not use these two variable, might be needed in future.
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

    def _predict_samples_native(self, fh, X=None):
        """Generate native Toto forecast object for a future horizon."""
        prediction_length = max(fh.to_relative(self._cutoff))

        forecast = self.model_.forecast(
            self._series,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
        )
        y_index = fh.to_absolute(self._cutoff)._values
        return forecast, y_index

    def _format_point_predictions(self, prediction_result, y_index, fh):
        """Format Toto point predictions."""
        if self.prediction_type.lower() == "median":
            all_predictions = prediction_result.median.cpu().squeeze(0).numpy().T
        else:
            all_predictions = prediction_result.mean.cpu().squeeze(0).numpy().T

        relative_indices = fh.to_relative(self._cutoff) - 1
        selected_predictions = all_predictions[relative_indices]

        return pd.DataFrame(
            selected_predictions, index=y_index, columns=self._y.columns
        )

    def _format_quantile_predictions(self, prediction_result, y_index, fh, alpha):
        """Format Toto quantile predictions."""
        import torch

        var_names = self._y.columns
        cols_idx = pd.MultiIndex.from_product([var_names, alpha])
        relative_indices = fh.to_relative(self._cutoff) - 1

        pred_quantiles = pd.DataFrame(index=y_index, columns=cols_idx)
        alpha_tensor = torch.tensor(alpha, device=self._device)

        quantiles = prediction_result.quantile(alpha_tensor)
        if quantiles.dim() > 3:
            quantile_values = quantiles.cpu().squeeze(1).numpy()
        else:
            quantile_values = quantiles.cpu().numpy()

        for i, var_name in enumerate(var_names):
            for j, a in enumerate(alpha):
                selected_quantiles = quantile_values[j, i, relative_indices]
                pred_quantiles[(var_name, a)] = selected_quantiles
        return pred_quantiles

    def _make_fallback_X(self, y):
        """Return no exogenous fallback for Toto."""
        return None

    def _load_native_model(self, model_path):
        """Load native Toto forecaster."""
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        toto_model = Toto.from_pretrained(**self.load_kwargs)
        toto_model.to(self._device)
        toto_model.compile()
        return TotoForecaster(toto_model.model)

    def _get_backend_device(self):
        """Return resolved Toto backend device."""
        return self._device

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        test_params = [
            {"num_samples": 2, "samples_per_batch": 2, "prediction_type": "median"},
            {"num_samples": 2, "samples_per_batch": 1, "prediction_type": "mean"},
            {"num_samples": 1, "samples_per_batch": 1, "prediction_type": "mean"},
        ]

        return test_params
