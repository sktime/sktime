"""Implements Chronos forecaster."""

__author__ = ["abdulfatir", "lostella", "Z-Fran", "benheid", "geetu040", "PranavBhatP"]
# abdulfatir and lostella for amazon-science/chronos-forecasting

__all__ = ["ChronosForecaster"]

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
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
    """Strategy for handling Chronos-Bolt and Chronos-2 models.

    Parameters
    ----------
    is_v2 : bool, default=False
        If True, the model is Chronos-2 (Chronos2Model architecture).
        Chronos-2 requires special loading via trust_remote_code=True because
        its config contains fields (input_patch_size, max_output_patches, etc.)
        that are unknown to sktime's local ChronosBoltConfig and ChronosConfig.
    """

    def __init__(self, is_v2: bool = False):
        self.is_v2 = is_v2

    def initialize_config(self) -> dict:
        return {
            "limit_prediction_length": False,
            "torch_dtype": torch.bfloat16,
            "device_map": "cpu",
        }

    def create_pipeline(self, key: str, kwargs: dict, use_source_package: bool):
        return _CachedChronosBolt(
            key=key,
            chronos_bolt_kwargs=kwargs,
            use_source_package=use_source_package,
            is_v2=self.is_v2,
        )

    def predict(
        self, pipeline, y_tensor: torch.Tensor, prediction_length: int, config: dict
    ) -> np.ndarray:
        # Create a copy of the config to avoid modifying the original
        predict_config = config.copy()

        # Strip Bolt-v1 / Chronos-2 incompatible parameters before calling predict
        predict_config.pop("max_output_patches", None)
        predict_config.pop("input_patch_size", None)

        prediction_results = pipeline.predict(
            y_tensor,
            prediction_length,
            limit_prediction_length=predict_config.get("limit_prediction_length", False),
        )
        return np.median(prediction_results[0].numpy(), axis=0)


class _Chronos2Pipeline:
    """Thin wrapper to load Chronos-2 via trust_remote_code.

    Chronos-2 ships its own model class (Chronos2Model) on HuggingFace with
    custom config fields (input_patch_size, max_output_patches, etc.) that are
    unknown to sktime's local ChronosConfig and ChronosBoltConfig.

    Loading via ``AutoModel`` with ``trust_remote_code=True`` defers config
    parsing to the model's own bundled code, bypassing the incompatible local
    config classes entirely.

    This class exposes a ``.predict()`` interface and a ``.config`` attribute
    compatible with what ``ChronosBoltStrategy.predict()`` and
    ``ChronosForecaster._predict()`` expect.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local path, e.g. ``"amazon/chronos-2"``.
    device_map : str, default="cpu"
        Device to load the model on. Use ``"cuda"`` for GPU, ``"mps"`` for
        Apple Silicon, or ``"cpu"`` for CPU inference.
    dtype : torch.dtype or None, default=None
        Torch dtype for model weights. Defaults to ``torch.bfloat16`` if None.
    """

    def __init__(self, model_path: str, device_map: str = "cpu", dtype=None):
        import torch
        from transformers import AutoConfig, AutoModel

        # Load the model config using the repo's own config class so that
        # Chronos-2 specific fields are parsed correctly.
        self.config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True
        )

        load_dtype = dtype if dtype is not None else torch.bfloat16

        # Load the full model using the repo's bundled modeling code.
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=load_dtype,
            device_map=device_map,
        )
        self._model.eval()

        self._dtype = load_dtype

    def predict(
        self,
        context: "torch.Tensor",
        prediction_length: int,
        limit_prediction_length: bool = False,
    ):
        """Run inference and return samples tensor.

        Parameters
        ----------
        context : torch.Tensor
            1-D tensor of historical time series values, shape (context_length,).
        prediction_length : int
            Number of future time steps to forecast.
        limit_prediction_length : bool, default=False
            Unused for Chronos-2; kept for interface compatibility.

        Returns
        -------
        tuple of torch.Tensor
            ``(samples,)`` where ``samples`` has shape
            ``(num_samples, prediction_length)``, matching the interface
            expected by ``ChronosBoltStrategy.predict()``.
        """
        import torch

        with torch.no_grad():
            device = next(self._model.parameters()).device
            # Model expects batch dimension: (batch=1, context_length)
            context_tensor = context.unsqueeze(0).to(
                dtype=self._dtype, device=device
            )
            output = self._model.generate(
                context=context_tensor,
                prediction_length=prediction_length,
            )

        # output shape: (batch=1, num_samples, prediction_length)
        # Return (samples,) where samples shape is (num_samples, prediction_length)
        return (output[0],)


class ChronosForecaster(BaseForecaster):
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
    model_pipeline: ChronosPipeline or ChronosBoltPipeline or _Chronos2Pipeline
        The underlying model pipeline used for forecasting.
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

    >>> # Example using 'amazon/chronos-2' model
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.chronos import ChronosForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> forecaster = ChronosForecaster("amazon/chronos-2")  # doctest: +SKIP
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
        "scitype:y": "univariate",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
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
        "device_map": "cpu",  # str
    }

    _default_chronos_bolt_config = {
        "limit_prediction_length": False,  # bool
        "torch_dtype": torch.bfloat16,  # torch.dtype
        "device_map": "cpu",  # str
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
        """Initialise model type and configuration based on model's architecture.

        Detects whether the model at ``self.model_path`` is a standard Chronos
        model, a Chronos-Bolt model, or a Chronos-2 model, and sets the
        appropriate strategy and default config accordingly.

        Chronos-2 (``Chronos2Model`` architecture) is routed through
        ``ChronosBoltStrategy(is_v2=True)``, which loads the model via
        ``_Chronos2Pipeline`` using ``trust_remote_code=True`` to avoid
        config-field incompatibilities with sktime's local pipeline classes.
        """
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            architectures = config.architectures or []

            is_bolt = "ChronosBoltModelForForecasting" in architectures
            is_v2 = "Chronos2Model" in architectures

            if is_bolt or is_v2:
                # Both Bolt and v2 use ChronosBoltStrategy; is_v2 controls
                # which loader is used inside _CachedChronosBolt.
                self.model_strategy = ChronosBoltStrategy(is_v2=is_v2)
                if is_v2:
                    print(
                        "🚀 Chronos v2 detected. Applying V2-Compatibility Strategy."
                    )
            else:
                self.model_strategy = ChronosDefaultStrategy()

            self._default_config = self.model_strategy.initialize_config()

            # Strip any Bolt-v1 fields that Chronos-2 doesn't support,
            # to avoid passing them downstream.
            if is_v2:
                self._default_config.pop("max_output_patches", None)
                self._default_config.pop("input_patch_size", None)

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
        self.model_pipeline = self._load_pipeline()
        return self

    def _get_chronos_kwargs(self):
        """Get the kwargs for Chronos model pipeline loader.

        Builds the keyword-argument dict passed to the pipeline's
        ``from_pretrained`` (or ``_Chronos2Pipeline.__init__``).
        Only keys understood by the target pipeline are included;
        Chronos-2-specific fields are never forwarded to the loader.
        """
        kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "device_map": self._config.get("device_map", "cpu"),
        }

        # transformers >= 4.x uses ``torch_dtype``; sktime's internal
        # ChronosPipeline.from_pretrained uses ``dtype``.  Pass both names
        # so either pipeline can pick up the correct key.
        if "torch_dtype" in self._config:
            # We use 'dtype' for the loader and remove 'torch_dtype' to stop the warning
            kwargs["dtype"] = self._config.pop("torch_dtype")

        return kwargs

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

    def __getstate__(self):
        """Return state for pickling, handling unpickleable model pipeline."""
        state = self.__dict__.copy()
        if hasattr(self, "model_pipeline"):
            state["model_pipeline"] = None
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state dictionary."""
        self.__dict__.update(state)

    def _ensure_model_pipeline_loaded(self):
        """Ensure model pipeline is loaded, recreating if needed after unpickling."""
        if not hasattr(self, "model_pipeline") or self.model_pipeline is None:
            if hasattr(self, "_is_fitted") and self._is_fitted:
                self.model_pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Load the model pipeline using the multiton pattern.

        Returns
        -------
        pipeline : ChronosPipeline or ChronosBoltPipeline or _Chronos2Pipeline
            The loaded model pipeline ready for predictions.
        """
        return self.model_strategy.create_pipeline(
            key=self._get_unique_chronos_key(),
            kwargs=self._get_chronos_kwargs(),
            use_source_package=self.use_source_package,
        ).load_from_checkpoint()

    def predict(self, fh=None, X=None, y=None):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted", i.e., ``self.is_fitted=True``.

        Accesses in self:

            * Fitted model attributes ending in "_".
            * ``self.cutoff``, ``self.is_fitted``

        Writes to self:
            Stores ``fh`` to ``self.fh`` if ``fh`` is passed and has not been passed
            previously.

        Parameters
        ----------
        fh : int, list, pd.Index coercible, or ``ForecastingHorizon``, default=None
            The forecasting horizon encoding the time stamps to forecast at.
            Should not be passed if has already been passed in ``fit``.
            If has not been passed in fit, must be passed, not optional

        X : time series in ``sktime`` compatible format, optional (default=None)
            Exogeneous time series to use in prediction.
            Should be of same scitype (``Series``, ``Panel``, or ``Hierarchical``)
            as ``y`` in ``fit``.
            If ``self.get_tag("X-y-must-have-same-index")``,
            ``X.index`` must contain ``fh`` index reference.
            If ``y`` is not passed (not performing global forecasting), ``X`` should
            only contain the time points to be predicted.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.

        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        y_pred : time series in sktime compatible data container format
            Point forecasts at ``fh``, with same index as ``fh``.
            ``y_pred`` has same type as the ``y`` that has been passed most recently:
            ``Series``, ``Panel``, ``Hierarchical`` scitype, same format (see above)

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        if self._fh is None and fh is not None:
            _fh = fh
        else:
            _fh = self._fh

        if y is not None:
            return self.fit_predict(fh=_fh, X=X, y=y)

        return super().predict(fh=fh, X=X)

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
        self._ensure_model_pipeline_loaded()

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
        _y = _y.values.reshape(1, -1, 1)

        results = []
        for i in range(_y.shape[0]):
            _y_i = _y[i, :, 0]

            # Resolve context_length safely across Chronos v1, Bolt, and v2.
            # v1/Bolt: config.context_length (int)
            # v2:      config.chronos_config["context_length"] (dict key)
            conf = self.model_pipeline.config
            c_len = getattr(conf, "context_length", None)
            if c_len is None and hasattr(conf, "chronos_config"):
                c_len = conf.chronos_config.get("context_length", 512)
            c_len = c_len or 512

            _y_i = _y_i[-c_len:]

            values = self.model_strategy.predict(
                self.model_pipeline,
                torch.Tensor(_y_i),
                prediction_length,
                self._config,
            )
            results.append(values)

        pred = np.stack(results, axis=1)

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

        y_pred = pred.loc[dateindex]
        return y_pred

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
        test_params.append(
            {
                "model_path": "amazon/chronos-bolt-tiny",
            }
        )
        test_params.append(
            {
                "model_path": "amazon/chronos-2",
            }
        )
        return test_params


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
    """Cached Chronos-Bolt / Chronos-2 model, one instance in memory.

    Chronos-Bolt is a zero-shot model and immutable, hence there will not be any
    side effects of sharing the same instance across multiple uses.

    For Chronos-2 (``is_v2=True``), loading is handled via
    ``_Chronos2Pipeline`` which uses ``trust_remote_code=True`` through
    ``transformers.AutoModel``.  This avoids crashing on config fields
    (``input_patch_size``, ``max_output_patches``, etc.) that are present in
    the Chronos-2 remote config but absent from sktime's local
    ``ChronosBoltConfig`` and ``ChronosConfig`` classes.

    Parameters
    ----------
    key : str
        Unique identifier used by the multiton cache.
    chronos_bolt_kwargs : dict
        Keyword arguments forwarded to the pipeline loader.
    use_source_package : bool
        If True, use the ``chronos`` source package instead of
        ``sktime.libs.chronos``.
    is_v2 : bool, default=False
        If True, load as Chronos-2 via ``_Chronos2Pipeline``.
    """

    def __init__(self, key, chronos_bolt_kwargs, use_source_package, is_v2=False):
        self.key = key
        self.chronos_bolt_kwargs = chronos_bolt_kwargs
        self.use_source_package = use_source_package
        self.is_v2 = is_v2
        self.model_pipeline = None

    def load_from_checkpoint(self):
        if self.model_pipeline is not None:
            return self.model_pipeline

        if self.is_v2:
            # Chronos-2 has a custom architecture that neither ChronosPipeline
            # nor ChronosBoltPipeline in sktime.libs.chronos can load without
            # crashing on unknown config fields.  Use _Chronos2Pipeline which
            # loads via AutoModel + trust_remote_code=True.
            if self.use_source_package:
                # The upstream chronos package may have native v2 support.
                try:
                    from chronos import ChronosBoltPipeline

                    self.model_pipeline = ChronosBoltPipeline.from_pretrained(
                        trust_remote_code=True,
                        **self.chronos_bolt_kwargs,
                    )
                except Exception:
                    # Fall back to our wrapper if the source package also fails.
                    self.model_pipeline = _Chronos2Pipeline(
                        model_path=self.chronos_bolt_kwargs[
                            "pretrained_model_name_or_path"
                        ],
                        device_map=self.chronos_bolt_kwargs.get("device_map", "cpu"),
                        dtype=self.chronos_bolt_kwargs.get("dtype", None),
                    )
            else:
                # sktime's local libs don't know about Chronos2Model config fields,
                # so always use our AutoModel-based wrapper.
                self.model_pipeline = _Chronos2Pipeline(
                    model_path=self.chronos_bolt_kwargs[
                        "pretrained_model_name_or_path"
                    ],
                    device_map=self.chronos_bolt_kwargs.get("device_map", "cpu"),
                    dtype=self.chronos_bolt_kwargs.get("dtype", None),
                )
        else:
            if self.use_source_package:
                from chronos import ChronosBoltPipeline
            else:
                from sktime.libs.chronos import ChronosBoltPipeline

            self.model_pipeline = ChronosBoltPipeline.from_pretrained(
                **self.chronos_bolt_kwargs,
            )

        return self.model_pipeline