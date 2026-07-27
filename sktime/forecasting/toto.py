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
    "siddharth7113",
]
__all__ = ["TotoForecaster"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)


class TotoForecaster(BaseFoundationForecaster):
    """Toto foundation model forecaster for zero-shot forecasting.

    Direct interface to forecaster from DataDog/toto [1]_.

    Toto is a foundation model for multivariate time series forecasting with a focus on
    observability metrics. This model leverages innovative architectural designs to
    efficiently handle the high-dimensional, complex time series that are characteristic
    of observability data. Generate both point forecasts and uncertainty estimates using
    a Student-T mixture model. Support for variable prediction horizons and context
    lengths.

    Known-future exogenous variables ``X`` are supported via Toto's native
    exogenous mechanism: the columns of ``X`` are appended after the target
    channels as exogenous variates. When ``X`` is used, it must be supplied for
    every step of the forecast horizon, i.e. for all steps ``1 .. max(fh)`` ahead
    of the cutoff (no gaps), since Toto consumes the known future values at each
    autoregressive step.

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

    With known-future exogenous variables:

    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> X, y = load_longley()
    >>> y_train, _, X_train, X_test = temporal_train_test_split(y, X, test_size=3)
    >>> model = TotoForecaster()
    >>> model.fit(y_train, X=X_train)  # doctest: +SKIP
    TotoForecaster()
    >>> forecast = model.predict(fh=[1, 2, 3], X=X_test)  # doctest: +SKIP
    """

    _tags = {
        "y_inner_mtype": ["pd.DataFrame"],
        "X_inner_mtype": ["pd.DataFrame"],
        "capability:multivariate": True,
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
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
            "siddharth7113",
        ],
        "maintainers": ["JATAYU000"],
        "python_version": ">= 3.10",
        "python_dependencies": ["torch>=2.5", "toto-ts>=0.1.3", "setuptools<82"],
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
            if not _check_soft_dependencies("xformers", severity="warning"):
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
        model_spec = FoundationModelSpec(
            model_path=model_path,
            device="auto" if device is None else device,
            random_state=seed,
            load_extra_kwargs={
                "use_memory_efficient_attention": use_memory_efficient_attention,
                "stabilize_with_global": stabilize_with_global,
                "scale_factor_exponent": scale_factor_exponent,
            },
            predict_extra_kwargs={
                "num_samples": num_samples,
                "samples_per_batch": samples_per_batch,
                "prediction_type": prediction_type,
            },
        )
        super().__init__(model_spec=model_spec)

    def __dynamic_tags__(self):
        """Set dependency tags for memory-efficient attention."""
        super().__dynamic_tags__()
        if self.use_memory_efficient_attention:
            self.set_tags(python_dependencies=["torch", "xformers", "accelerate"])

    def _update_attrs_in_fit(self, y, X=None, fh=None):
        """Convert the fitted target context to Toto's native container."""
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        device = self.model_spec.device
        if X is not None:
            combined = pd.concat([y, X], axis=1)
            input_series = torch.tensor(
                combined.values.T,
                dtype=torch.float32,
                device=device,
            )
            self._num_exog_ = X.shape[1]
        else:
            input_series = torch.tensor(
                y.values.T,
                dtype=torch.float32,
                device=device,
            )
            self._num_exog_ = 0

        self._n_targets_ = y.shape[1]
        id_mask = torch.zeros_like(input_series)
        padding_mask = torch.full_like(input_series, True, dtype=torch.bool)

        # current model does not use these two variable, might be needed in future.
        timestamp_seconds = torch.zeros_like(input_series)
        time_interval_seconds = torch.full(
            (input_series.shape[0],), 60 * 15, dtype=torch.float32
        ).to(device)

        self._series = MaskedTimeseries(
            series=input_series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
            num_exogenous_variables=self._num_exog_,
        )

    def _load_model(self):
        """Load the Toto model and forecaster."""
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        model_spec = self.model_spec
        toto_model = Toto.from_pretrained(
            pretrained_model_name_or_path=model_spec.model_path,
            **model_spec.load_extra_kwargs,
        )
        toto_model.to(model_spec.device)
        toto_model.compile()
        forecaster = TotoForecaster(toto_model.model)
        return ModelHandle(model=toto_model, pipeline=forecaster)

    def _inference(
        self,
        handle,
        context_y,
        context_X,
        future_X,
        pred_len,
        fh,
        alpha=None,
    ):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
            If not passed in _fit, guaranteed to be passed here
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        model_spec = self.model_spec
        predict_kwargs = model_spec.predict_extra_kwargs
        future_exog = self._build_future_exog(future_X, pred_len)
        forecast = handle.pipeline.forecast(
            self._series,
            prediction_length=pred_len,
            num_samples=predict_kwargs["num_samples"],
            samples_per_batch=predict_kwargs["samples_per_batch"],
            future_exogenous_variables=future_exog,
        )
        if predict_kwargs["prediction_type"].lower() == "median":
            all_predictions = forecast.median.cpu().squeeze(0).numpy().T
            point_result = {"median": all_predictions}
        else:
            all_predictions = forecast.mean.cpu().squeeze(0).numpy().T
            point_result = {"mean": all_predictions}

        point_result = {
            key: values[:, : self._n_targets_] for key, values in point_result.items()
        }

        quantile_results = None
        if alpha is not None:
            import torch

            alpha_tensor = torch.tensor(alpha, device=model_spec.device)
            quantiles = forecast.quantile(alpha_tensor)
            if quantiles.dim() > 3:
                quantile_values = quantiles.cpu().squeeze(1).numpy()
            else:
                quantile_values = quantiles.cpu().numpy()
            quantile_results = {
                value: quantile_values[i].T[:, : self._n_targets_]
                for i, value in enumerate(alpha)
            }

        return ForecastResult(
            **point_result,
            quantiles=quantile_results,
        )

    def _build_future_exog(self, X, prediction_length):
        """Build the future exogenous tensor for Toto's ``forecast`` call.

        Toto rolls out a contiguous block of ``prediction_length`` steps and, for
        each step, replaces the exogenous channels with the known future values.
        It therefore needs ``X`` for **every** step ``1 .. prediction_length``
        ahead of the cutoff, shaped ``(batch, num_exogenous, future_time_steps)``.

        Parameters
        ----------
        X : pd.DataFrame or None
            Future exogenous values passed to ``predict``.
        prediction_length : int
            Number of contiguous steps Toto will forecast.

        Returns
        -------
        torch.Tensor or None
            Tensor of shape ``(1, num_exogenous, prediction_length)`` if the
            forecaster was fitted with exogenous variables, else ``None``.
        """
        if self._num_exog_ == 0:
            return None

        import torch

        from sktime.forecasting.base import ForecastingHorizon

        if X is None:
            raise ValueError(
                "TotoForecaster was fitted with exogenous variables X, so X must "
                "also be passed to predict, covering the full forecast horizon."
            )

        # contiguous absolute index for steps 1 .. prediction_length
        full_fh = ForecastingHorizon(range(1, prediction_length + 1), is_relative=True)
        future_index = full_fh.to_absolute(self._cutoff)._values

        # align user X onto every step Toto rolls through; gaps become NaN
        X_future = X.reindex(future_index)
        if X_future.isnull().values.any():
            raise ValueError(
                "TotoForecaster requires exogenous X for every step in the "
                "forecast horizon. Provide X covering all steps from 1 to "
                f"{prediction_length} ahead of the cutoff (no gaps)."
            )

        # shape (time, n_exog) -> (n_exog, time) -> (1, n_exog, time)
        future_exog = (
            torch.tensor(X_future.values.T, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.model_spec.device)
        )
        return future_exog

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
            {
                "seed": 42,
                "num_samples": 2,
                "samples_per_batch": 2,
                "prediction_type": "median",
            },
            {
                "seed": 42,
                "num_samples": 2,
                "samples_per_batch": 1,
                "prediction_type": "mean",
            },
            {
                "seed": 42,
                "num_samples": 1,
                "samples_per_batch": 1,
                "prediction_type": "mean",
            },
        ]

        return test_params
