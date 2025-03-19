"""Implements Chronos forecaster."""

__author__ = ["abdulfatir", "lostella", "Z-Fran", "benheid", "geetu040", "PranavBhatP"]
# abdulfatir and lostella for amazon-science/chronos-forecasting

__all__ = ["ChronosForecaster"]

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.utils.singleton import _multiton

if _check_soft_dependencies("torch", severity="none"):
    import torch
else:

    class torch:
        """Dummy class if torch is unavailable."""

        bfloat16 = None

        class Tensor:
            """Dummy class if torch is unavailable."""


if _check_soft_dependencies("transformers", severity="none"):
    import transformers
else:

    class PreTrainedModel:
        """Dummy class if transformers is unavailable."""


class ChronosModelStrategy(ABC):
    """Abstract base class defining the interface for Chronos model strategies."""

    @abstractmethod
    def initialize_config(self) -> dict:
        """Initialise the default configuration of the model."""
        pass

    @abstractmethod
    def create_pipeline(self, key: str, kwargs: dict, use_source_package: bool):
        """Create the appropriate pipeline for the model.

        This method handles the creation of a cached pipeline instance for th specific
        model type (Chronos or Chronos-bolt).

        Parameters
        ----------
        key: str
            Unique identifier for the model instance.
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

    def create_pipeline(self, key: str, kwargs: dict, use_source_package: bool):
        return _CachedChronos(
            key=key, chronos_kwargs=kwargs, use_source_package=use_source_package
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

    def create_pipeline(self, key: str, kwargs: dict, use_source_package: bool):
        return _CachedChronosBolt(
            key=key, chronos_bolt_kwargs=kwargs, use_source_package=use_source_package
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


class ChronosForecaster(_BaseGlobalForecaster):
    """
    Interface to the Chronos and Chronos-Bolt Zero-Shot Forecaster by Amazon Research.

    Chronos and Chronos-Bolt are pretrained time-series foundation models
    developed by Amazon for time-series forecasting. This method has been
    proposed in [2]_ and official code is given at [1]_.

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
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "scitype:y": "univariate",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
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
        seed: Optional[int] = None,
        use_source_package: bool = False,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.use_source_package = use_source_package
        self.ignore_deps = ignore_deps

        # set random seed
        self.seed = seed
        self._seed = np.random.randint(0, 2**31) if seed is None else seed

        # initialize model_strategy as None, will be set correctly after loading config.
        self.model_strategy = None

        # set config
        self.config = config
        self._config = None

        self.context = None

        if self.ignore_deps:
            self.set_tags(python_dependencies=[])
        elif self.use_source_package:
            self.set_tags(python_dependencies=["chronos"])
        else:
            self.set_tags(python_dependencies=["torch", "transformers", "accelerate"])

        super().__init__()

        self._initialize_model_type()

    def _initialize_model_type(self):
        """Intialise model type and configuration based on model's architecture."""
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(self.model_path)

            # "ChronosBoltModelForForecasting is the name of the architecture"
            # as specified in the config.json file
            is_bolt = "ChronosBoltModelForForecasting" in (config.architectures or [])

            if is_bolt:
                self.model_strategy = ChronosBoltStrategy()
            else:
                self.model_strategy = ChronosDefaultStrategy()

            self._default_config = self.model_strategy.initialize_config()
            self._config = self._default_config.copy()
            if self.config is not None:
                self._config.update(self.config)

        except Exception as e:
            raise ValueError(
                f"Failed to load model configuration from {self.model_path}. "
                f"Error: {str(e)}"
            ) from e

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : reference to self
        """
        self.model_pipeline = self.model_strategy.create_pipeline(
            key=self._get_unique_chronos_key(),
            kwargs=self._get_chronos_kwargs(),
            use_source_package=self.use_source_package,
        ).load_from_checkpoint()
        return self

    def _get_chronos_kwargs(self):
        """Get the kwargs for Chronos model."""
        return {
            "pretrained_model_name_or_path": self.model_path,
            "torch_dtype": self._config["torch_dtype"],
            "device_map": self._config["device_map"],
        }

    def _get_unique_chronos_key(self):
        """Get unique key for Chronos model to use in multiton."""
        model_path = self.model_path
        use_source_package = self.use_source_package
        kwargs = self._get_chronos_kwargs()
        kwargs_plus_model_path = {
            **kwargs,
            "model_path": model_path,
            "use_source_package": use_source_package,
        }
        return str(sorted(kwargs_plus_model_path.items()))

    def _predict(self, fh, y=None, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : pd.DataFrame
            Predicted forecasts.
        """
        transformers.set_seed(self._seed)
        if fh is not None:
            # needs to be integer not np.int64
            prediction_length = int(max(fh.to_relative(self.cutoff)))
        else:
            prediction_length = 1

        _y = self._y.copy()
        if y is not None:
            _y = y.copy()
        _y_df = _y

        index_names = _y.index.names
        if isinstance(_y.index, pd.MultiIndex):
            _y = _frame2numpy(_y)
        else:
            _y = _y.values.reshape(1, -1, 1)

        results = []
        for i in range(_y.shape[0]):
            _y_i = _y[i, :, 0]
            _y_i = _y_i[-self.model_pipeline.model.config.context_length :]

            values = self.model_strategy.predict(
                self.model_pipeline, torch.Tensor(_y_i), prediction_length, self._config
            )
            results.append(values)

        pred = np.stack(results, axis=1)
        if isinstance(_y_df.index, pd.MultiIndex):
            ins = np.array(
                list(np.unique(_y_df.index.droplevel(-1)).repeat(pred.shape[0]))
            )
            ins = [ins[..., i] for i in range(ins.shape[-1])] if ins.ndim > 1 else [ins]

            idx = (
                ForecastingHorizon(range(1, pred.shape[0] + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values.tolist()
                * pred.shape[1]
            )
            index = pd.MultiIndex.from_arrays(
                ins + [idx],
                names=_y_df.index.names,
            )
        else:
            index = (
                ForecastingHorizon(range(1, pred.shape[0] + 1))
                .to_absolute(self._cutoff)
                ._values
            )
        pred_out = fh.get_expected_pred_idx(_y, cutoff=self.cutoff)

        pred = pd.DataFrame(
            pred.reshape(-1, 1),
            index=index,
            columns=_y_df.columns,
        )
        dateindex = pred.index.get_level_values(-1).map(lambda x: x in pred_out)
        pred.index.names = index_names

        return pred.loc[dateindex]

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


def _same_index(data):
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(lambda x: x.equals(data.iloc[0])).all(), (
        "All series must has the same index"
    )
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data):
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr


@_multiton
class _CachedChronos:
    """Cached Chronos model, to ensure only one instance exists in memory.

    Chronos is a zero shot model and immutable, hence there will not be
    any side effects of sharing the same instance across multiple uses.
    """

    def __init__(self, key, chronos_kwargs, use_source_package):
        self.key = key
        self.chronos_kwargs = chronos_kwargs
        self.use_source_package = use_source_package
        self.model_pipeline = None

    def load_from_checkpoint(self):
        if self.model_pipeline is not None:
            return self.model_pipeline

        if self.use_source_package:
            from chronos import ChronosPipeline
        else:
            from sktime.libs.chronos import ChronosPipeline

        self.model_pipeline = ChronosPipeline.from_pretrained(
            **self.chronos_kwargs,
        )

        return self.model_pipeline


@_multiton
class _CachedChronosBolt:
    """Cached Chronos-Bolt model, to ensure only one instance exists in memory.

    Chronos-Bolt is a zero-shot model and immutable, hence there will not be any
    side effects of sharing the same instance across multiple uses.
    """

    def __init__(self, key, chronos_bolt_kwargs, use_source_package):
        self.key = key
        self.chronos_bolt_kwargs = chronos_bolt_kwargs
        self.use_source_package = use_source_package
        self.model_pipeline = None

    def load_from_checkpoint(self):
        if self.model_pipeline is not None:
            return self.model_pipeline

        if self.use_source_package:
            from chronos import ChronosBoltPipeline
        else:
            from sktime.libs.chronos import ChronosBoltPipeline

        self.model_pipeline = ChronosBoltPipeline.from_pretrained(
            **self.chronos_bolt_kwargs,
        )

        return self.model_pipeline
