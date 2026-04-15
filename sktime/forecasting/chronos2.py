"""Implements Chronos-2 forecaster."""

__author__ = ["priyanshuharshbodhi1"]

__all__ = ["Chronos2Forecaster"]

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


if _check_soft_dependencies("transformers", severity="none"):
    import transformers
else:

    class transformers:
        """Dummy class if transformers is unavailable."""

        @staticmethod
        def set_seed(seed):
            """Set random seed."""


class Chronos2Forecaster(BaseForecaster):
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

    Attributes
    ----------
    model_pipeline : Chronos2Pipeline
        The underlying model pipeline used for forecasting.

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
        "authors": ["priyanshuharshbodhi1"],
        "maintainers": ["priyanshuharshbodhi1"],
        "python_dependencies": ["chronos-forecasting>=2.0.0"],
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": False,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
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
        "torch_dtype": torch.bfloat16,
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
        self.seed = seed
        self._seed = np.random.randint(0, 2**31) if seed is None else seed
        self.config = config
        self.ignore_deps = ignore_deps

        self._config = self._default_config.copy()
        if config is not None:
            self._config.update(config)

        self.model_pipeline = None

        if ignore_deps:
            self.set_tags(python_dependencies=[])

        super().__init__()

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable model pipeline."""
        state = self.__dict__.copy()
        if hasattr(self, "model_pipeline"):
            state["model_pipeline"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    def _get_pipeline_kwargs(self):
        return {
            "pretrained_model_name_or_path": self.model_path,
            "torch_dtype": self._config["torch_dtype"],
            "device_map": self._config["device_map"],
        }

    def _get_unique_key(self):
        kwargs = self._get_pipeline_kwargs()
        return str(sorted(kwargs.items()))

    def _load_pipeline(self):
        return _CachedChronos2(
            key=self._get_unique_key(),
            chronos2_kwargs=self._get_pipeline_kwargs(),
        ).load_from_checkpoint()

    def _ensure_model_pipeline_loaded(self):
        """Reload model pipeline if needed after unpickling."""
        if not hasattr(self, "model_pipeline") or self.model_pipeline is None:
            if hasattr(self, "_is_fitted") and self._is_fitted:
                self.model_pipeline = self._load_pipeline()

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series.
        X : pd.DataFrame, optional
            Past exogenous covariates.
        fh : ForecastingHorizon, optional

        Returns
        -------
        self
        """
        self.model_pipeline = self._load_pipeline()
        return self

    def predict(self, fh=None, X=None, y=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, pd.Index or ForecastingHorizon, optional
        X : time series in sktime compatible format, optional
            Future exogenous covariates.
        y : time series in sktime compatible format, optional
            Historical values for global forecasting. If provided,
            performs fit_predict on the new series.

        Returns
        -------
        y_pred : time series in sktime compatible format
        """
        if self._fh is None and fh is not None:
            _fh = fh
        else:
            _fh = self._fh

        if y is not None:
            return self.fit_predict(fh=_fh, X=X, y=y)

        return super().predict(fh=fh, X=X)

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
        X : pd.DataFrame, optional
            Future exogenous covariates (known-future). Column names must be
            a subset of X provided in fit.

        Returns
        -------
        y_pred : pd.DataFrame
        """
        self._ensure_model_pipeline_loaded()
        transformers.set_seed(self._seed)

        prediction_length = int(max(fh.to_relative(self.cutoff)))

        context_length = self._config["context_length"]
        if context_length is None:
            context_length = self.model_pipeline.model_context_length

        _y = self._y.copy()
        y_vals = _y.values.T

        if y_vals.shape[1] > context_length:
            y_vals = y_vals[:, -context_length:]

        input_dict = {"target": y_vals}

        if self._X is not None:
            actual_len = y_vals.shape[1]
            past_X = self._X.values[-actual_len:]
            input_dict["past_covariates"] = {
                col: past_X[:, i] for i, col in enumerate(self._X.columns)
            }

        if X is not None:
            if self._X is None:
                raise ValueError(
                    "X was not provided in fit but is provided in predict. "
                    "To use future covariates, provide past covariate values "
                    "in fit as well."
                )
            future_vals = X.values[:prediction_length]
            input_dict["future_covariates"] = {
                col: future_vals[:, i] for i, col in enumerate(X.columns)
            }

        predictions = self.model_pipeline.predict(
            [input_dict],
            prediction_length=prediction_length,
            batch_size=self._config["batch_size"],
            context_length=context_length,
            cross_learning=self._config["cross_learning"],
            limit_prediction_length=self._config["limit_prediction_length"],
        )

        pred_tensor = predictions[0]
        quantiles = self.model_pipeline.quantiles
        median_idx = quantiles.index(0.5)
        point_forecast = pred_tensor[:, median_idx, :].numpy()

        index = (
            ForecastingHorizon(range(1, prediction_length + 1))
            .to_absolute(self._cutoff)
            ._values
        )
        pred_out = fh.get_expected_pred_idx(self._y, cutoff=self.cutoff)

        pred_df = pd.DataFrame(
            point_forecast.T,
            index=index,
            columns=self._y.columns,
        )
        pred_df.index.names = self._y.index.names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"model_path": "amazon/chronos-2"},
            {"model_path": "amazon/chronos-2", "seed": 42},
        ]


@_multiton
class _CachedChronos2:
    """Cached Chronos-2 model to ensure only one instance exists in memory.

    Chronos-2 is a zero-shot model and immutable, so sharing the same instance
    across multiple uses has no side effects.
    """

    def __init__(self, key, chronos2_kwargs):
        self.key = key
        self.chronos2_kwargs = chronos2_kwargs
        self.model_pipeline = None

    def load_from_checkpoint(self):
        """Load Chronos-2 pipeline from pretrained checkpoint."""
        if self.model_pipeline is not None:
            return self.model_pipeline

        from chronos import Chronos2Pipeline

        self.model_pipeline = Chronos2Pipeline.from_pretrained(**self.chronos2_kwargs)
        return self.model_pipeline
