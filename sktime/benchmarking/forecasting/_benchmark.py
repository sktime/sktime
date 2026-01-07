# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Forecasting model benchmarking framework.

This module provides utilities for benchmarking multiple forecasting models
on simulated or real time series data.
"""

__author__ = ["sktime developers"]
__all__ = ["ForecastingBenchmark"]


import numpy as np
import pandas as pd

from sktime.base import BaseEstimator
from sktime.forecasting.base import BaseForecaster
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def _root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _weighted_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Weighted Mean Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


from sktime.registry import all_estimators
from sktime.utils.validation.forecasting import check_fh


class ForecastingBenchmark(BaseEstimator):
    """Benchmark multiple forecasting models on time series data.

    The ForecastingBenchmark class provides a framework for testing and comparing
    multiple forecasting models on specified time series data. It supports both
    simulated and real data, and can automatically select univariate-capable models
    from sktime's registry or use user-specified models.

    Parameters
    ----------
    models : list of tuples or None, default=None
        List of models to benchmark. Each model can be specified as:
        - Tuple: (name, model_instance) where model_instance is an initialized
          forecaster with custom parameters, e.g.,
          ("auto_arima", AutoARIMA(sp=12, suppress_warnings=True))
        - Just model_instance: Will use class name as model name
        If None, all univariate forecasters from sktime registry will be used
        (excluding those with soft dependencies not installed).
    fh : int, list or ForecastingHorizon, default=1
        Forecasting horizon to use for predictions.
    metrics : list of str or callable, default=None
        Metrics to calculate for evaluation. If None, uses
        ["mae", "mse", "mape"]. Can be:
        - str: Name of metric from sktime.performance_metrics.forecasting
        - callable: Custom metric function with signature (y_true, y_pred)
    test_size : int or float, default=0.2
        If int, number of observations to use as test set.
        If float, proportion of data to use as test set.
    cv_folds : int, optional, default=None
        Number of cross-validation folds. If None, uses simple train/test split.
    n_jobs : int, default=1
        Number of jobs to run in parallel for model fitting. -1 means using all
        processors.
    verbose : bool, default=True
        Whether to print progress information.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    results_ : pd.DataFrame
        DataFrame containing benchmarking results with columns for each metric
        and rows for each model.
    fitted_models_ : dict
        Dictionary mapping model names to fitted model instances.
    predictions_ : dict
        Dictionary mapping model names to their predictions.
    errors_ : dict
        Dictionary mapping model names to any errors encountered during fitting.

    Examples
    --------
    >>> from sktime.benchmarking.forecasting import (
    ...     TimeSeriesSimulator,
    ...     ForecastingBenchmark
    ... )
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>>
    >>> # Generate synthetic data
    >>> sim = TimeSeriesSimulator(
    ...     length=100,
    ...     distribution="poisson",
    ...     dist_params={"lam": 10}
    ... )
    >>> y = sim.simulate()
    >>>
    >>> # Benchmark models with custom parameters
    >>> models = [
    ...     ("naive_mean", NaiveForecaster(strategy="mean")),
    ...     ("naive_drift", NaiveForecaster(strategy="drift")),
    ...     ("theta", ThetaForecaster(sp=12))
    ... ]
    >>> benchmark = ForecastingBenchmark(
    ...     models=models,
    ...     fh=[1, 2, 3],
    ...     test_size=20
    ... )
    >>> results = benchmark.run(y)
    >>> print(results)  # doctest: +SKIP
    >>> best_model = benchmark.get_best_model("mae")  # doctest: +SKIP
    """

    _tags = {
        "authors": "sktime developers",
        "maintainers": "sktime developers",
        "object_type": "benchmarker",
        "python_dependencies": None,
    }

    # Default metrics - ordered by most commonly used
    _DEFAULT_METRICS = {
        "mae": mean_absolute_error,
        "rmse": _root_mean_squared_error,
        "mse": mean_squared_error,
        "mape": mean_absolute_percentage_error,
        "wmape": _weighted_mean_absolute_percentage_error,
    }

    def __init__(
        self,
        models=None,
        fh=1,
        metrics=None,
        test_size=0.2,
        cv_folds=None,
        n_jobs=1,
        verbose=True,
        random_state=None,
    ):
        if isinstance(test_size, float) and not (0 < test_size < 1):
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")
        if isinstance(test_size, int) and test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")

        self.models = models
        self.fh = fh
        self.metrics = metrics
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        super().__init__()

    def run(self, y, X=None):
        """Run benchmark on the provided time series.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Target time series to forecast.
        X : pd.DataFrame, optional, default=None
            Exogenous variables.

        Returns
        -------
        results : pd.DataFrame
            Benchmark results with metrics for each model.
        """
        # Initialize results storage
        self.results_ = None
        self.fitted_models_ = {}
        self.predictions_ = {}
        self.errors_ = {}

        # Get models to benchmark
        models_to_test = self._get_models_to_test(y)

        if len(models_to_test) == 0:
            raise ValueError(
                "No models available for benchmarking. "
                "Please specify models explicitly or install additional dependencies."
            )

        if self.verbose:
            print(f"Benchmarking {len(models_to_test)} models...")

        # Split data
        y_train, y_test = self._split_data(y)

        # Get forecasting horizon
        fh = check_fh(self.fh)

        # Get metrics
        metrics_dict = self._get_metrics()

        # Run benchmark for each model
        results_list = []
        for model_name, model in models_to_test:
            if self.verbose:
                print(f"  Testing {model_name}...")

            try:
                # Fit and predict
                model.fit(y_train, X=X, fh=fh)
                y_pred = model.predict(fh=fh, X=X)

                # Store fitted model and predictions
                self.fitted_models_[model_name] = model
                self.predictions_[model_name] = y_pred

                model_results = {"model": model_name}
                for metric_name, metric_func in metrics_dict.items():
                    try:
                        score = metric_func(y_test[: len(y_pred)], y_pred)
                        model_results[metric_name] = score
                    except Exception as e:
                        model_results[metric_name] = np.nan
                        if self.verbose:
                            print(
                                f"    Warning: Could not calculate {metric_name}: {e}"
                            )

                results_list.append(model_results)

            except Exception as e:
                self.errors_[model_name] = str(e)
                if self.verbose:
                    print(f"    Error: {e}")

                model_results = {"model": model_name}
                for metric_name in metrics_dict.keys():
                    model_results[metric_name] = np.nan
                results_list.append(model_results)

        # Create results DataFrame
        self.results_ = pd.DataFrame(results_list)
        self.results_ = self.results_.set_index("model")

        if self.verbose:
            print("\n" + "=" * 70)
            print("BENCHMARKING RESULTS")
            print("=" * 70)
            print(f"\nTested {len(results_list)} models on {len(y)} observations")
            print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
            print(f"Forecast horizon: {fh}")
            print("\nPerformance Metrics (lower is better):")
            print("-" * 70)
            # Sort by first metric for easier comparison
            first_metric = list(metrics_dict.keys())[0]
            sorted_results = self.results_.sort_values(by=first_metric)
            print(sorted_results.to_string())
            print("-" * 70)
            print(f"\nBest model (by {first_metric}): {sorted_results.index[0]}")
            print(f"Best {first_metric}: {sorted_results.iloc[0][first_metric]:.4f}")
            if len(self.errors_) > 0:
                print(f"\nWarning: {len(self.errors_)} model(s) failed to fit")
            print("=" * 70)

        return self.results_

    def _get_models_to_test(self, y):
        """Get list of models to test.

        Parameters
        ----------
        y : pd.Series
            Time series to check for compatibility.

        Returns
        -------
        models_list : list of tuple
            List of (name, model) tuples.
        """
        if self.models is not None:
            # Use user-specified models
            models_list = []
            for item in self.models:
                if isinstance(item, tuple):
                    name, model = item
                elif isinstance(item, BaseForecaster):
                    name = item.__class__.__name__
                    model = item
                else:
                    raise ValueError(
                        "Models must be BaseForecaster instances or "
                        "(name, model) tuples."
                    )
                models_list.append((name, model))
        else:
            # Get all univariate forecasters from registry
            try:
                all_forecasters = all_estimators(
                    estimator_types="forecaster",
                    filter_tags={"scitype:y": "univariate"},
                    return_names=True,
                    as_dataframe=False,
                )

                models_list = []
                for name, ForecasterClass in all_forecasters:
                    try:
                        model = ForecasterClass.create_test_instance()
                        models_list.append((name, model))
                    except Exception:  # noqa: S110
                        # Skip models with missing dependencies
                        pass

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load models from registry: {e}")
                models_list = []

        return models_list

    def _split_data(self, y):
        """Split time series into train and test sets.

        Parameters
        ----------
        y : pd.Series
            Time series to split.

        Returns
        -------
        y_train : pd.Series
            Training set.
        y_test : pd.Series
            Test set.
        """
        if isinstance(self.test_size, float):
            split_point = int(len(y) * (1 - self.test_size))
        else:
            split_point = len(y) - self.test_size

        if split_point <= 0 or split_point >= len(y):
            raise ValueError(
                f"Invalid train/test split. Data length: {len(y)}, "
                f"test_size: {self.test_size}, split_point: {split_point}"
            )

        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]

        return y_train, y_test

    def _get_metrics(self):
        """Get metrics dictionary.

        Returns
        -------
        metrics_dict : dict
            Dictionary mapping metric names to metric functions.
        """
        if self.metrics is None:
            return self._DEFAULT_METRICS.copy()

        metrics_dict = {}
        for metric in self.metrics:
            if isinstance(metric, str):
                if metric in self._DEFAULT_METRICS:
                    metrics_dict[metric] = self._DEFAULT_METRICS[metric]
                else:
                    raise ValueError(
                        f"Unknown metric: {metric}. "
                        f"Available metrics: {list(self._DEFAULT_METRICS.keys())}"
                    )
            elif callable(metric):
                # Custom metric function
                metric_name = getattr(metric, "__name__", "custom_metric")
                metrics_dict[metric_name] = metric
            else:
                raise ValueError("Metrics must be strings or callable functions.")

        return metrics_dict

    def get_best_model(self, metric="mae", lower_is_better=True):
        """Get the best performing model based on a specific metric.

        Parameters
        ----------
        metric : str, default="mae"
            Metric to use for selection.
        lower_is_better : bool, default=True
            Whether lower metric values indicate better performance.

        Returns
        -------
        best_model_name : str
            Name of the best performing model.
        best_model : BaseForecaster
            The fitted best performing model.
        best_score : float
            Best score achieved.
        """
        if self.results_ is None:
            raise ValueError("Must run benchmark before getting best model.")

        if metric not in self.results_.columns:
            raise ValueError(
                f"Metric '{metric}' not found in results. "
                f"Available metrics: {list(self.results_.columns)}"
            )

        # Get best model
        if lower_is_better:
            best_idx = self.results_[metric].idxmin()
        else:
            best_idx = self.results_[metric].idxmax()

        best_model_name = best_idx
        best_model = self.fitted_models_[best_model_name]
        best_score = self.results_.loc[best_model_name, metric]

        return best_model_name, best_model, best_score

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from sktime.forecasting.naive import NaiveForecaster

        params1 = {
            "models": [("naive", NaiveForecaster())],
            "fh": 1,
            "test_size": 10,
            "verbose": False,
        }
        params2 = {
            "models": [("naive", NaiveForecaster())],
            "fh": [1, 2, 3],
            "metrics": ["mae", "mse"],
            "test_size": 0.2,
            "verbose": False,
        }
        return [params1, params2]
