"""Implements Chronos-2 forecaster."""

__all__ = ["Chronos2Forecaster"]

from sktime.forecasting.foundation._base2 import BaseFoundationForecaster
from sktime.forecasting.foundation._result import ForecastResult, ModelHandle


class Chronos2Forecaster(BaseFoundationForecaster):
    """Interface to the Chronos-2 Zero-Shot Forecaster by Amazon Research.

    Chronos-2 is a pretrained encoder-only time series foundation model
    developed by Amazon for zero-shot forecasting. It supports univariate,
    multivariate, and covariate-informed forecasting tasks within a single
    architecture. The official code and technical report are given at [1]_ and [2]_.

    Unlike Chronos (v1), Chronos-2 natively handles multivariate targets,
    past-only covariates, and known-future covariates via a group attention
    mechanism described in [2]_.

    Parameters
    ----------
    model_path : str, default="amazon/chronos-2"
        Path to the Chronos-2 HuggingFace model.

    config : dict, optional, default=None
        Configuration overrides. Supported keys:

        - "limit_prediction_length" : bool, default=False
            If True, raises an error when prediction_length exceeds the model's
            maximum prediction length.
        - "torch_dtype" : torch.dtype, default=torch.bfloat16
            Data type for model weights and operations.
        - "device_map" : str, default="cpu"
            Device for inference, e.g., "cpu", "cuda", or "mps".
        - "batch_size" : int, default=256
            Number of time series per batch during prediction.
        - "context_length" : int or None, default=None
            Maximum context length for inference. Defaults to model's
            context length (8192 for amazon/chronos-2).
        - "cross_learning" : bool, default=False
            If True, enables cross-learning across all input series in a batch,
            sharing information via the group attention mechanism.

    seed : int or None, optional, default=None
        Random seed for reproducibility.

    ignore_deps : bool, optional, default=False
        If True, dependency checks are skipped.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting
    .. [2] Abdul Fatir Ansari and others (2025).
       Chronos-2: Towards a Universal, General-Purpose Forecasting Foundation Model.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.chronos2 import Chronos2Forecaster
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> forecaster = Chronos2Forecaster("amazon/chronos-2")  # doctest: +SKIP
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": ["priyanshuharshbodhi1", "fkiraly"],
        "maintainers": ["priyanshuharshbodhi1"],
        "python_dependencies": ["chronos-forecasting>=2.0.0"],
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "capability:multivariate": True,
        "capability:insample": False,
        "capability:global_forecasting": True,
        "capability:non_contiguous_X": False,
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    }

    _default_config = {
        "limit_prediction_length": False,
        "torch_dtype": "torch.bfloat16",
        "device_map": "cpu",
        "batch_size": 256,
        "context_length": None,
        "cross_learning": False,
    }

    def __init__(
        self,
        model_path: str = "amazon/chronos-2",
        config: dict = None,
        seed: int | None = None,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.config = config
        self.seed = seed
        self.ignore_deps = ignore_deps

        normalized_config = self._default_config.copy()
        if config is not None:
            normalized_config.update(config)

        self.limit_prediction_length = normalized_config["limit_prediction_length"]
        self.batch_size = normalized_config["batch_size"]
        self.context_length = normalized_config["context_length"]
        self.cross_learning = normalized_config["cross_learning"]

        super().__init__(
            model_path=model_path,
            device=normalized_config["device_map"],
            dtype=normalized_config["torch_dtype"],
            random_state=seed,
            ignore_deps=ignore_deps,
        )

    def _load_model(self):
        """Load a Chronos-2 checkpoint into a cacheable model handle."""
        from chronos import Chronos2Pipeline

        model = Chronos2Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=self.dtype_,
            device_map=self.device_,
        )
        return ModelHandle(model=model)

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
        """Build Chronos-2 context and return its median forecast."""
        model = handle.model

        context_length = self.context_length
        if context_length is None:
            context_length = model.model_context_length

        context = context_y.iloc[-context_length:]
        target = context.to_numpy().T
        input_dict = {"target": target}

        if context_X is not None:
            actual_len = target.shape[1]
            past_X = context_X.to_numpy()[-actual_len:]
            input_dict["past_covariates"] = {
                col: past_X[:, i] for i, col in enumerate(context_X.columns)
            }

        if future_X is not None:
            if context_X is None:
                raise ValueError(
                    "X was not provided in fit but is provided in predict. "
                    "To use future covariates, provide past covariate values "
                    "in fit as well."
                )
            future_vals = future_X.to_numpy()[:pred_len]
            input_dict["future_covariates"] = {
                col: future_vals[:, i] for i, col in enumerate(future_X.columns)
            }

        predictions = model.predict(
            [input_dict],
            prediction_length=pred_len,
            batch_size=self.batch_size,
            context_length=context_length,
            cross_learning=self.cross_learning,
            limit_prediction_length=self.limit_prediction_length,
        )

        pred_tensor = predictions[0]
        median_idx = model.quantiles.index(0.5)
        point_forecast = pred_tensor[:, median_idx, :].detach().cpu().numpy()
        return ForecastResult(median=point_forecast.T, raw=predictions)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"model_path": "amazon/chronos-2"},
            {"model_path": "amazon/chronos-2", "seed": 42},
        ]
