"""Implements Chronos forecaster."""

__author__ = ["abdulfatir", "lostella", "Z-Fran", "benheid", "geetu040", "PranavBhatP"]
# abdulfatir and lostella for amazon-science/chronos-forecasting

__all__ = ["ChronosForecaster"]

from abc import ABC, abstractmethod

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _GlobalForecastingDeprecationMixin
from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)

if _check_soft_dependencies("torch", severity="none"):
    import torch
else:

    class torch:
        """Dummy class if torch is unavailable."""

        bfloat16 = None

        class Tensor:
            """Dummy class if torch is unavailable."""


class ChronosModelStrategy(ABC):
    """Abstract base class defining the interface for Chronos model strategies."""

    @abstractmethod
    def initialize_config(self) -> dict:
        """Initialise the default configuration of the model."""
        pass

    @abstractmethod
    def create_pipeline(self, kwargs: dict, use_source_package: bool):
        """Create the appropriate pipeline for the model.

        This method handles the creation of a cached pipeline instance for th specific
        model type (Chronos or Chronos-bolt).

        Parameters
        ----------
        kwargs: dict
            Configuration parameters for the model pipeline. Should include:

            - pretrained_model_name_or_path : str
                Path to the pretrained model
            - torch_dtype : torch.dtype
                Data type for model computations
            - device_map : str
                Device to run the model on ('cpu', 'cuda', etc.)
        use_source_package: bool
            If True, uses the original chronos package.
            If False, uses the sktime implementation.

        Returns
        -------
        _CachedChronos or _CachedChronosBolt
            A cached instance of the appropriate pipeline class that can be used to load
            the model checkpoint.
        """
        pass

    @abstractmethod
    def predict(
        self, pipeline, y_tensor: torch.Tensor, predictions_length: int, config: dict
    ) -> dict:
        """Make predictions using the model pipeline.

        Parameters
        ----------
        pipeline : ChronosPipeline or ChronosBoltPipeline
            The initialized model pipeline for making predictions.
        y_tensor : torch.Tensor
            Input time series data as a PyTorch tensor.
        prediction_length : int
            Number of future time steps to predict.
        config : dict
            Configuration dictionary containing model-specific parameters.
            For Chronos models, this includes:
                - num_samples : int or None
                    Number of samples to generate for prediction.
                - temperature : float or None
                    Sampling temperature for predictions.
                - top_k : int or None
                    Limits sampling to top k predictions.
                - top_p : float or None
                    Cumulative probability threshold for nucleus sampling.
                - limit_prediction_length : bool
                    Whether to limit prediction length to model's context length.
            For Chronos-Bolt models, this includes:
                - limit_prediction_length : bool
                    Whether to limit prediction length to model's context length.

        Returns
        -------
        np.ndarray
            Array containing the predicted values. For Chronos models, shape is
            (prediction_length,). For Chronos-Bolt models, shape is
            (prediction_length,).
        """
        pass


class ChronosDefaultStrategy(ChronosModelStrategy):
    """Strategy for handling standard set of Chronos Models."""

    def initialize_config(self) -> dict:
        return {
            "num_samples": None,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "limit_prediction_length": False,
            "torch_dtype": torch.bfloat16,
            "device_map": "cpu",
        }

    def create_pipeline(self, kwargs: dict, use_source_package: bool):
        if use_source_package:
            from chronos import ChronosPipeline
        else:
            from sktime.libs.chronos import ChronosPipeline

        return ChronosPipeline.from_pretrained(
            **kwargs,
        )

    def predict(
        self, pipeline, y_tensor: torch.Tensor, prediction_length: int, config: dict
    ) -> np.ndarray:
        prediction_results = pipeline.predict(
            y_tensor,
            prediction_length,
            num_samples=config["num_samples"],
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            limit_prediction_length=config["limit_prediction_length"],
        )
        return np.median(prediction_results[0].numpy(), axis=0)


class ChronosBoltStrategy(ChronosModelStrategy):
    """Strategy for handling Chronos-Bolt models."""

    def initialize_config(self) -> dict:
        return {
            "limit_prediction_length": False,
            "torch_dtype": torch.bfloat16,
            "device_map": "cpu",
        }

    def create_pipeline(self, kwargs: dict, use_source_package: bool):
        if use_source_package:
            from chronos import ChronosBoltPipeline
        else:
            from sktime.libs.chronos import ChronosBoltPipeline

        return ChronosBoltPipeline.from_pretrained(
            **kwargs,
        )

    def predict(
        self, pipeline, y_tensor: torch.Tensor, prediction_length: int, config: dict
    ) -> np.ndarray:
        prediction_results = pipeline.predict(
            y_tensor,
            prediction_length,
            limit_prediction_length=config["limit_prediction_length"],
        )
        return np.median(prediction_results[0].numpy(), axis=0)


class ChronosForecaster(_GlobalForecastingDeprecationMixin, BaseFoundationForecaster):
    """
    Interface to the Chronos and Chronos-Bolt Zero-Shot Forecaster by Amazon Research.

    Chronos and Chronos-Bolt are pretrained time-series foundation models
    developed by Amazon for time-series forecasting. This method has been
    proposed in [2]_ and official code is given at [1]_.

    Note: vanilla Chronos is not exogenous capable despite being so advertised in [2]_.
    The "exogenous capable" version is actually a composite forecaster rather than
    an exogenous capable foundation model.

    To obtain this "exogenous capable" version of Chronos as advertised in [2]_,
    combine ``ChronosForecaster`` with an exogenous capable forecaster via
    ``ResidualBoostingForecaster``. The original reference uses
    tabularized linear regression, i.e., ``YtoX(LinearRegression())``,
    with ``YtoX`` from ``sktime`` and ``LinearRegression`` from ``sklearn``.

    Parameters
    ----------
    model_path : str
        Path to the Chronos huggingface model.

    config : dict, optional, default={}
        A dictionary specifying the configuration settings for the model.
        The available configuration options include hyperparameters that control
        the prediction behavior, sampling, and hardware preferences. In case of the
        ``Chronos`` model, the dictionary can include the following keys:

        - "num_samples" : int, optional
            The number of samples to generate during prediction. Median of these samples
            is taken to get prediction for each timestamp.
        - "temperature" : float, optional
            Sampling temperature for prediction. A higher value increases the randomness
            of predictions, while a lower value makes them more deterministic.
        - "top_k" : int, optional
            Limits the sampling pool to the top k predictions during sampling.
        - "top_p" : float, optional
            Cumulative probability threshold for nucleus sampling.
            Controls the diversity of the predictions.

        The below configuration options are available in both model options:
        - "limit_prediction_length" : bool, default=False
            If True, limits the length of the predictions to the model's context length.
        - "torch_dtype" : torch.dtype, default=torch.bfloat16
            Data type to use for model weights and operations (e.g., `torch.float32`,
            `torch.float16`, or `torch.bfloat16`).
        - "device_map" : str, default="cpu"
            Specifies the device on which to run the model, e.g.,
            "cpu" for CPU inference, "cuda" for GPU, or "mps" for Apple Silicon.

        If not provided, the default values from the pretrained model or system
        configuration are used.

    seed: int, optional, default=None
        Random seed for transformers.

    use_source_package: bool, optional, default=False
        If True, the model will be loaded directly from the source package ``chronos``.
        This is useful if you want to bypass the local version of the package
        or when working in an environment where the latest updates
        from the source package are needed.
        If False, the model will be loaded from the local version of package maintained
        in sktime.
        To install the source package, follow the instructions here [1]_.

    ignore_deps: bool, optional, default=False
        If True, dependency checks will be ignored, and the user is expected to handle
        the installation of required packages manually. If False, the class will enforce
        the default dependencies required for Chronos.

    Attributes
    ----------
    model_pipeline: ChronosPipeline or ChronosBoltPipeline
        The underlying model pipeline user for forecasting
    is_bolt: bool
        Indicates whether the model is a Chronos-Bolt model, to ensure
        effective differentiation purely from model-path.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting
    .. [2] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, and others (2024).
    Chronos: Learning the Language of Time Series

    Examples
    --------
    >>> # Example using 'amazon/chronos-t5-tiny' model
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.chronos import ChronosForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> forecaster = ChronosForecaster("amazon/chronos-t5-tiny")  # doctest: +SKIP
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh)  # doctest: +SKIP

    >>> # Example using 'amazon/chronos-bolt-tiny' model
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.chronos import ChronosForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> forecaster = ChronosForecaster("amazon/chronos-bolt-tiny")  # doctest: +SKIP
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh)  # doctest: +SKIP
    """

    # tag values are "safe defaults" which can usually be left as-is
    _tags = {
        # packaging info
        # --------------
        "authors": [
            "abdulfatir",
            "lostella",
            "Z-Fran",
            "benheid",
            "geetu040",
            "rigvedmanoj",
        ],
        # abdulfatir and lostella for amazon-science/chronos-forecasting
        "maintainers": ["geetu040"],
        "python_dependencies": ["torch", "transformers", "accelerate"],
        # estimator type
        # --------------
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": False,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        "capability:unequal_length": False,
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:libs": ["sktime.libs.chronos"],
        "tests:skip_by_name": [  # pickling problems
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    }

    _default_chronos_config = {
        "num_samples": None,  # int, use value from pretrained model if None
        "temperature": None,  # float, use value from pretrained model if None
        "top_k": None,  # int, use value from pretrained model if None
        "top_p": None,  # float, use value from pretrained model if None
        "limit_prediction_length": False,  # bool
        "torch_dtype": torch.bfloat16,  # torch.dtype
        "device_map": "cpu",  # str, use "cpu" for CPU inference, "cuda" for gpu and "mps" for Apple Silicon # noqa
    }

    _default_chronos_bolt_config = {
        "limit_prediction_length": False,  # bool
        "torch_dtype": torch.bfloat16,  # torch.dtype
        "device_map": "cpu",  # str, use "cpu" for CPU inference, "cuda" for gpu and "mps" for Apple Silicon # noqa
    }

    def __init__(
        self,
        model_path: str,
        config: dict = None,
        seed: int | None = None,
        use_source_package: bool = False,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.config = config
        self.use_source_package = use_source_package
        self.seed = seed
        self.ignore_deps = ignore_deps

        normalized_config = self._default_chronos_config.copy()
        if config is not None:
            normalized_config.update(config)

        model_spec = FoundationModelSpec(
            model_path=model_path,
            device=normalized_config["device_map"],
            dtype=normalized_config["torch_dtype"],
            random_state=seed,
            ignore_deps=ignore_deps,
            load_extra_kwargs={"use_source_package": use_source_package},
        )
        super().__init__(model_spec=model_spec)

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        super().__dynamic_tags__()
        if self.model_spec.ignore_deps:
            return
        if self.use_source_package:
            self.set_tags(python_dependencies=["chronos"])
        else:
            self.set_tags(python_dependencies=["torch", "transformers", "accelerate"])

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        super().__post_init__()

        # initialize model_strategy as None, will be set correctly after loading config.
        self.model_strategy = None

        self._initialize_model_type()

    def _initialize_model_type(self):
        """Initialise model type and configuration based on model's architecture."""
        from transformers import AutoConfig

        spec = self.model_spec_
        try:
            config = AutoConfig.from_pretrained(spec.model_path)

            # "ChronosBoltModelForForecasting is the name of the architecture"
            # as specified in the config.json file
            is_bolt = "ChronosBoltModelForForecasting" in (config.architectures or [])

            if is_bolt:
                self.model_strategy = ChronosBoltStrategy()
            else:
                self.model_strategy = ChronosDefaultStrategy()

            predict_kwargs = self.model_strategy.initialize_config()
            if self.config is not None:
                load_config_names = {"device_map", "torch_dtype"}
                predict_kwargs.update(
                    {
                        key: value
                        for key, value in self.config.items()
                        if key not in load_config_names
                    }
                )
            self._update_model_spec(predict_extra_kwargs=predict_kwargs)

        except Exception as e:
            raise ValueError(
                f"Failed to load model configuration from {spec.model_path}. "
                f"Error: {str(e)}"
            ) from e

    def _load_model(self):
        """Load the model pipeline."""
        model_spec = self.model_spec_
        pipeline = self.model_strategy.create_pipeline(
            kwargs={
                "pretrained_model_name_or_path": model_spec.model_path,
                "torch_dtype": model_spec.dtype,
                "device_map": model_spec.device,
            },
            use_source_package=model_spec.load_extra_kwargs["use_source_package"],
        )
        return ModelHandle(model=pipeline.model, pipeline=pipeline)

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
        """Make predictions using the model pipeline."""
        model_spec = self.model_spec_
        _y = context_y.values.reshape(1, -1, 1)

        results = []
        for i in range(_y.shape[0]):
            _y_i = _y[i, :, 0]
            _y_i = _y_i[-handle.pipeline.model.config.context_length :]

            values = self.model_strategy.predict(
                handle.pipeline,
                torch.Tensor(_y_i),
                pred_len,
                model_spec.predict_extra_kwargs,
            )
            results.append(values)

        pred = np.stack(results, axis=1)
        return ForecastResult(median=pred)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        test_params = []
        test_params.append(
            {
                "model_path": "amazon/chronos-t5-tiny",
            }
        )
        test_params.append(
            {
                "model_path": "amazon/chronos-t5-tiny",
                "config": {
                    "num_samples": 20,
                },
                "seed": 42,
            }
        )
        test_params.append(
            {
                "model_path": "amazon/chronos-bolt-tiny",
            }
        )
        return test_params
