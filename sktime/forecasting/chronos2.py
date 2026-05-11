"""Implements Chronos-2 forecaster."""

__all__ = ["Chronos2Forecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.singleton import _multiton


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
        "authors": ["priyanshuharshbodhi1", "fkiraly"],
        "maintainers": ["priyanshuharshbodhi1"],
        "python_dependencies": ["chronos-forecasting>=2.0.0"],
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": False,
        "capability:missing_values": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
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
        self.config = config
        self.ignore_deps = ignore_deps

        self.model_pipeline = None

        if ignore_deps:
            self.set_tags(python_dependencies=[])

        super().__init__()

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * dynamic tag setting
        * any soft dependency imports in the constructor
        """
        self._seed = np.random.randint(0, 2**31) if self.seed is None else self.seed

        import torch

        self._config = self._default_config.copy()
        self._config["torch_dtype"] = torch.bfloat16

        if self.config is not None:
            self._config.update(self.config)

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

        context_length = self._config["context_length"]
        if context_length is None:
            context_length = self.model_pipeline.model_context_length

        context = y

        if context.shape[0] > context_length:
            context = context.iloc[-context_length:]

        context = context.values.T

        self._context = context
        self._y_index_names = y.index.names
        return self

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
        import transformers

        self._ensure_model_pipeline_loaded()
        transformers.set_seed(self._seed)

        prediction_length = int(max(fh.to_relative(self.cutoff)))

        context_length = self._config["context_length"]
        if context_length is None:
            context_length = self.model_pipeline.model_context_length

        context = self._context
        input_dict = {"target": context}

        if self._X is not None:
            actual_len = context.shape[1]
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
        pred_out = fh.get_expected_pred_idx(context, cutoff=self.cutoff)

        pred_df = pd.DataFrame(
            point_forecast.T,
            index=index,
            columns=self._get_varnames(),
        )
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return quantile forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional
            Future exogenous covariates.
        alpha : list of float, optional
            A list of probabilities at which quantile forecasts are computed.
            If None, uses the model's default quantiles.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
            second level being the quantile forecasts for each alpha.
            Row index is fh, with additional (upper) levels equal to instance levels,
            from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
        """
        import transformers

        self._ensure_model_pipeline_loaded()
        transformers.set_seed(self._seed)

        prediction_length = int(max(fh.to_relative(self.cutoff)))

        context_length = self._config["context_length"]
        if context_length is None:
            context_length = self.model_pipeline.model_context_length

        context = self._context
        input_dict = {"target": context}

        if self._X is not None:
            actual_len = context.shape[1]
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
        model_quantiles = self.model_pipeline.quantiles

        # Use model's quantiles if alpha is None
        if alpha is None:
            alpha = model_quantiles
        
        # Map requested alphas to model quantile indices
        quantile_indices = []
        for a in alpha:
            if a not in model_quantiles:
                raise ValueError(
                    f"Requested quantile {a} not available in model. "
                    f"Available quantiles: {model_quantiles}"
                )
            quantile_indices.append(model_quantiles.index(a))

        # Extract quantile predictions
        quantile_forecasts = pred_tensor[:, quantile_indices, :].numpy()

        # Build index for forecasts
        index = (
            ForecastingHorizon(range(1, prediction_length + 1))
            .to_absolute(self._cutoff)
            ._values
        )
        pred_out = fh.get_expected_pred_idx(context, cutoff=self.cutoff)

        # Build multi-index columns: (variable, quantile)
        var_names = self._get_varnames()
        columns = pd.MultiIndex.from_product(
            [var_names, alpha],
            names=["variable", "quantile"]
        )

        # Reshape: (prediction_length, n_quantiles * n_variables)
        # Each variable's quantiles should be consecutive
        n_vars = len(var_names)
        n_quantiles = len(alpha)
        reshaped = np.zeros((prediction_length, n_vars * n_quantiles))
        
        for i, var_idx in enumerate(range(n_vars)):
            for j, q_idx in enumerate(quantile_indices):
                col_idx = i * n_quantiles + j
                reshaped[:, col_idx] = quantile_forecasts[var_idx, j, :]

        pred_df = pd.DataFrame(
            reshaped,
            index=index,
            columns=columns,
        )
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    def _predict_interval(self, fh, X=None, coverage=None):
        """Compute/return interval forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional
            Future exogenous covariates.
        coverage : list of float, optional
            Nominal coverage(s) of predictive interval(s).
            If None, uses [0.9] (90% coverage).

        Returns
        -------
        intervals : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
            second level coverage fractions, third level is "lower" or "upper".
            Row index is fh, with additional (upper) levels equal to instance levels,
            from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
        """
        if coverage is None:
            coverage = [0.9]

        # Convert coverage to quantiles
        alphas_for_intervals = []
        for c in coverage:
            lower_alpha = (1 - c) / 2
            upper_alpha = 1 - lower_alpha
            alphas_for_intervals.extend([lower_alpha, upper_alpha])

        # Get quantile predictions
        quantile_pred = self._predict_quantiles(fh=fh, X=X, alpha=alphas_for_intervals)

        # Restructure to interval format
        var_names = self._get_varnames()
        
        # Build multi-index columns: (variable, coverage, bound)
        columns = pd.MultiIndex.from_product(
            [var_names, coverage, ["lower", "upper"]],
            names=["variable", "coverage", "bound"]
        )

        interval_data = []
        for var in var_names:
            for c in coverage:
                lower_alpha = (1 - c) / 2
                upper_alpha = 1 - lower_alpha
                
                lower_vals = quantile_pred[(var, lower_alpha)].values
                upper_vals = quantile_pred[(var, upper_alpha)].values
                
                interval_data.append(lower_vals)
                interval_data.append(upper_vals)

        interval_df = pd.DataFrame(
            np.column_stack(interval_data),
            index=quantile_pred.index,
            columns=columns,
        )

        return interval_df

    def _predict_proba(self, fh, X=None, marginal=True):
        """Compute/return fully probabilistic forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional
            Future exogenous covariates.
        marginal : bool, optional
            Whether returned distribution is marginal by time index.

        Returns
        -------
        pred_dist : skpro BaseDistribution
            Predictive distribution at fh.
        """
        from skpro.distributions.empirical import Empirical

        # Get all available quantiles from the model
        quantile_pred = self._predict_quantiles(fh=fh, X=X, alpha=None)
        
        # Extract the quantiles and reshape for Empirical distribution
        # Empirical expects shape (n_samples, n_vars) per time point
        var_names = self._get_varnames()
        model_quantiles = self.model_pipeline.quantiles
        
        # For each time point, create an empirical distribution
        # We'll use the quantiles as sample points with equal weights
        n_timepoints = len(quantile_pred)
        n_vars = len(var_names)
        n_quantiles = len(model_quantiles)
        
        # Reshape quantile predictions into samples
        # Shape: (n_timepoints, n_quantiles, n_vars)
        samples = np.zeros((n_timepoints, n_quantiles, n_vars))
        
        for i, var in enumerate(var_names):
            for j, q in enumerate(model_quantiles):
                samples[:, j, i] = quantile_pred[(var, q)].values
        
        # Create Empirical distribution
        # spl parameter expects (n_samples, n_vars) for each time point
        # We'll pass the transposed samples
        pred_dist = Empirical(
            spl=samples,
            index=quantile_pred.index,
            columns=pd.Index(var_names)
        )
        
        return pred_dist
    
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
