"""Implements Chronos 2 forecaster."""

__all__ = ["Chronos2Forecaster"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
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


class Chronos2Forecaster(BaseForecaster):
    """Interface to the Chronos 2 Zero-Shot Forecaster by Amazon Research.

    Chronos 2 is a pretrained time series foundation model developed by Amazon
    for zero-shot forecasting. Unlike Chronos v1, it natively supports
    multivariate forecasting, exogenous covariates (past and future), and
    quantile-based probabilistic predictions. This method has been proposed
    in [2]_ and official code is given at [1]_.

    Parameters
    ----------
    model_path : str, default="amazon/chronos-2"
        Path to the Chronos 2 HuggingFace model. Available models include
        ``"amazon/chronos-2"`` (120M parameters).
    config : dict, optional, default={}
        A dictionary specifying the configuration settings for the model.
        The available configuration options include:
        - "batch_size" : int, default=256
        Batch size for inference.
        - "context_length" : int or None, default=None
        Maximum context length. If None, uses the model default (8192).
        - "cross_learning" : bool, default=False
        If True, enables cross-learning across variates for multivariate forecasting.
        - "limit_prediction_length" : bool, default=False
        If True, limits the prediction length to the model's maximum.
        - "torch_dtype" : torch.dtype, default=torch.bfloat16
        Data type to use for model weights and operations.
        - "device_map" : str, default="cpu"
        Specifies the device on which to run the model, example:
        "cpu" for CPU inference, "cuda" for GPU, or "mps" for Apple Silicon.
    quantile_levels : list of float, default=[0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9]
        Quantile levels for probabilistic predictions.
    seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    model_pipeline : Chronos2Pipeline
        The underlying model pipeline used for forecasting.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting
    .. [2] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, and others (2025).
       Chronos 2: From Univariate to Universal Forecasting

    """

    # tag values are "safe defaults" which can usually be left as-is
    _tags = {
        # packaging info
        # --------------
        "authors": [],
        # abdulfatir and lostella for amazon-science/chronos-forecasting
        "maintainers": [""],
        "python_dependencies": ["torch", "transformers", "accelerate"],
        # estimator type
        # --------------
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:pred_int": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:libs": [],
        "tests:skip_by_name": [],
    }
    _default_config = {
        "batch_size": 256,
        "context_length": None,
        "cross_learning": False,
        "limit_prediction_length": False,
        "torch_dtype": torch.bfloat16,
        "device_map": "cpu",
    }

    def __init__(
        self,
        model_path: str = "amazon/chronos-2",
        # there is a small version of the model,
        # "amazon/chronos-2-small" (20M parameters),
        # Should it be supported as well??
        config: dict = None,
        quantile_levels: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        seed: int | None = None,
    ):
        self.model_path = model_path
        self.quantile_levels = quantile_levels

        # set random seed for reproducibility
        self.seed = seed
        self._seed = np.random.randint(0, 2**31) if seed is None else seed

        self.config = config
        self._config = self._default_config.copy()
        if config is not None:
            self._config.update(config)

        self.context = None

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            The target time series to which the forecaster will be fitted.
            Can be univariate (Series) or multivariate (DataFrame).
        X : pd.DataFrame, optional, default=None
            Exogenous covariates corresponding to the target series. Should
            have the same index as `y` and columns for each covariate.
        fh : ForecastingHorizon, optional, default=None
            The forecasting horizon with the time points to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self.model_pipeline = self._load_pipeline()
        return self

    def _get_chronos2_kwargs(self):
        """Get Chronos 2 specific kwargs from the config."""
        # This method would extract and validate the relevant configuration
        # settings for the Chronos 2 model, such as batch size, context length,
        # cross-learning flag, etc, and return them in a format suitable for
        # initializing the model pipeline.
        """Get the kwargs for Chronos model."""
        return {
            "pretrained_model_name_or_path": self.model_path,
            "torch_dtype": self._config["torch_dtype"],
            "device_map": self._config["device_map"],
        }

    def _get_unique_chronos2_key(self):
        """Get unique key for Chronos model to use in multiton."""
        model_path = self.model_path
        kwargs = self._get_chronos2_kwargs()
        kwargs_plus_model_path = {
            **kwargs,
            "model_path": model_path,
        }
        return str(sorted(kwargs_plus_model_path.items()))

    def _ensure_model_pipeline_loaded(self):
        """Ensure model pipeline is loaded, recreating if needed after unpickling."""
        if not hasattr(self, "model_pipeline") or self.model_pipeline is None:
            if hasattr(self, "_is_fitted") and self._is_fitted:
                self.model_pipeline = self._load_pipeline()
            else:
                raise RuntimeError(
                    "Model pipeline not loaded. Fit the forecaster before predicting."
                )

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.

        X : pd.DataFrame, optional (default=None)
            Future covariates corresponding to the forecasting horizon. Should
            have an index that matches the absolute time points in `fh`
            and columns for each covariate.All future covariate keys
            must be a subset of past covariate keys of past covariates are provided.

        Returns
        -------
        y_pred : pd.DataFrame
            Point forecasts at the time points specified by `fh`.
            Columns correspond to variates if multivariate, and index corresponds to
            the forecasted time points.
        """
        self._ensure_model_pipeline_loaded()
        transformers.set_seed(self._seed)
        # If no past covariates then only y
        # else concatenate past covariates and y to form the context for prediction
        if self._X is None:
            self.context = self._y.copy()
        else:
            self.context = pd.concat([self._y, self._X], axis=1)
        # for predict_df we need to pass the context and future covariates seperately

        # configuring the context
        # adding id_column if we have multiple timeseries
        if "id" not in self.context.columns:
            self.context["id"] = "series_1"
        # adding timestamp
        if "timestamp" not in self.context.columns:
            self.context["timestamp"] = self.context.index
        # configuring the future covariates
        if X is not None:
            if "id" not in X.columns:
                X["id"] = "series_1"
            if "timestamp" not in X.columns:
                X["timestamp"] = X.index

        pred_out = self.model_pipeline.predict_df(
            self.context,
            future_df=X,
            prediction_length=len(self._fh),
            id_column="id",
            timestamp_column="timestamp",
            target=self._y.columns.tolist(),
            batch_size=self._config["batch_size"],
            context_length=self._config["context_length"],
            cross_learning=self._config["cross_learning"],
        )

        # to transfrom the output into same format as the input y
        pred = pred_out.pivot(
            index="timestamp", columns="target_name", values="predictions"
        ).sort_index()
        # remove the names of index and columns to match the format of input y
        pred.index.name = None
        pred.columns.name = None

        return pred

    def _load_pipeline(self):
        """Load the model pipeline using the multiton pattern.

        Returns
        -------
        pipeline : Chronos2Pipeline
            The loaded model pipeline ready for predictions.
        """
        return _CachedChronos2(
            key=self._get_unique_chronos2_key(),
            chronos2_kwargs=self._get_chronos2_kwargs(),
        ).load_from_checkpoint()


@_multiton
class _CachedChronos2:
    """Cached Chronos 2 model, to ensure only one instance exists in memory.

    Chronos 2 is a zero-shot model and immutable, hence there will not be
    any side effects of sharing the same instance across multiple uses.
    """

    def __init__(self, key, chronos2_kwargs):
        self.key = key
        self.chronos2_kwargs = chronos2_kwargs
        self.model_pipeline = None

    def load_from_checkpoint(self):
        if self.model_pipeline is not None:
            return self.model_pipeline

        from chronos import Chronos2Pipeline

        self.model_pipeline = Chronos2Pipeline.from_pretrained(
            **self.chronos2_kwargs,
        )

        return self.model_pipeline


# if support for chronos 2 small is added,
# it needs to cached separately
# dummy comment for [skip ci] commit
