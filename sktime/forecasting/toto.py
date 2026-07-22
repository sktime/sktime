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

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
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
        super().__init__(model_path=model_path, device=device, random_state=seed)

    def __dynamic_tags__(self):
        """Set dependency tags for memory-efficient attention."""
        super().__dynamic_tags__()
        if self.use_memory_efficient_attention:
            self.set_tags(python_dependencies=["torch", "xformers", "accelerate"])

    def _resolve_device(self):
        """Resolve Toto's automatic CUDA-or-CPU device policy."""
        if self.device is not None:
            return self.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_toto_kwargs(self):
        """Get keyword arguments for the Toto model.

        Returns
        -------
        dict
            Keyword arguments for the Toto model.
        """
        return {
            "pretrained_model_name_or_path": self.model_path,
            "use_memory_efficient_attention": self.use_memory_efficient_attention,
            "stabilize_with_global": self.stabilize_with_global,
            "scale_factor_exponent": self.scale_factor_exponent,
        }

    def _update_attrs_in_fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.

            * if self.get_tag("capability:multivariate")==False:
              guaranteed to be univariate (e.g., single-column for DataFrame)
            * if self.get_tag("capability:multivariate")==True: no restrictions apply,
              the method should handle uni- and multivariate y appropriately

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        input_series = torch.tensor(y.values.T, dtype=torch.float32).to(self.device_)

        id_mask = torch.zeros_like(input_series).to(self.device_)
        padding_mask = torch.full_like(input_series, True, dtype=torch.bool).to(
            self.device_
        )

        # current model does not use these two variable, might be needed in future.
        timestamp_seconds = torch.zeros_like(input_series)
        time_interval_seconds = torch.full(
            (input_series.shape[0],), 60 * 15, dtype=torch.float32
        ).to(self.device_)

        self._series = MaskedTimeseries(
            series=input_series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

    def _load_model(self):
        """Load the Toto model and forecaster."""
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        toto_model = Toto.from_pretrained(**self._get_toto_kwargs())
        toto_model.to(self.device_)
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
        forecast = handle.pipeline.forecast(
            self._series,
            prediction_length=pred_len,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
        )
        if self.prediction_type.lower() == "median":
            all_predictions = forecast.median.cpu().squeeze(0).numpy().T
            point_result = {"median": all_predictions}
        else:
            all_predictions = forecast.mean.cpu().squeeze(0).numpy().T
            point_result = {"mean": all_predictions}

        quantile_results = None
        if alpha is not None:
            import torch

            alpha_tensor = torch.tensor(alpha, device=self.device_)
            quantiles = forecast.quantile(alpha_tensor)
            if quantiles.dim() > 3:
                quantile_values = quantiles.cpu().squeeze(1).numpy()
            else:
                quantile_values = quantiles.cpu().numpy()
            quantile_results = {
                value: quantile_values[i].T for i, value in enumerate(alpha)
            }

        return ForecastResult(
            **point_result,
            quantiles=quantile_results,
        )

    def _cache_key_extra(self):
        """Return model-loading parameters specific to Toto."""
        return tuple(
            sorted(
                (key, value)
                for key, value in self._get_toto_kwargs().items()
                if key != "pretrained_model_name_or_path"
            )
        )

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
