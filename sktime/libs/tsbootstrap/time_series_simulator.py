from numbers import Integral
from typing import Any

import numpy as np
from numpy.random import Generator

from tsbootstrap.tsfit import TSFit
from tsbootstrap.utils.types import ModelTypes
from tsbootstrap.utils.validate import (
    validate_fitted_model,
    validate_integers,
    validate_literal_type,
    validate_rng,
    validate_X_and_y,
)


class TimeSeriesSimulator:
    """
    Class to simulate various types of time series models.

    Attributes
    ----------
    n_samples: int
        Number of samples in the fitted time series model.
    n_features: int
        Number of features in the fitted time series model.
    burnin: int
        Number of burn-in samples to discard for certain models.

    Methods
    -------
    _validate_ar_simulation_params(params)
        Validate the parameters necessary for the simulation.
    _simulate_ar_residuals(lags, coefs, init, max_lag)
        Simulates an Autoregressive (AR) process with given lags, coefficients, initial values, and random errors.
    simulate_ar_process(resids_lags, resids_coefs, resids)
        Simulate AR process from the fitted model.
    _simulate_non_ar_residuals()
        Simulate residuals according to the model type.
    simulate_non_ar_process()
        Simulate a time series from the fitted model.
    generate_samples_sieve(model_type, resids_lags, resids_coefs, resids)
        Generate a bootstrap sample using the sieve bootstrap.
    """

    _tags = {"python_dependencies": ["arch", "statsmodels"]}

    def __init__(
        self,
        fitted_model,
        X_fitted: np.ndarray,
        rng=None,
    ) -> None:
        """
        Initialize the TimeSeriesSimulator class.

        Parameters
        ----------
        fitted_model: FittedModelTypes
            A fitted model object.
        X_fitted: np.ndarray
            Array of fitted values.
        rng: Optional[Union[Integral, Generator]], optional
            Random number generator instance. Defaults to None.
        """
        self.fitted_model = fitted_model
        self.X_fitted = X_fitted
        self.rng = rng
        self.n_samples, self.n_features = self.X_fitted.shape
        self.burnin = min(100, self.n_samples // 3)

    @property
    def fitted_model(self):
        """Get the fitted model."""
        return self._fitted_model

    @fitted_model.setter
    def fitted_model(self, fitted_model) -> None:
        """Set the fitted model, ensuring it's validated first."""
        validate_fitted_model(fitted_model)
        self._fitted_model = fitted_model

    @property
    def X_fitted(self) -> np.ndarray:
        """Get the array of fitted values."""
        return self._X_fitted

    @X_fitted.setter
    def X_fitted(self, value: np.ndarray) -> None:
        """
        Set the array of fitted values.

        Parameters
        ----------
        value: np.ndarray
            Array of fitted values to set.
        """
        from arch.univariate.base import ARCHModelResult
        from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

        model_is_var = isinstance(self.fitted_model, VARResultsWrapper)
        model_is_arch = isinstance(self.fitted_model, ARCHModelResult)
        self._X_fitted, _ = validate_X_and_y(
            value, None, model_is_var=model_is_var, model_is_arch=model_is_arch
        )

    @property
    def rng(self):
        """Get the random number generator instance."""
        return self._rng

    @rng.setter
    def rng(self, rng) -> None:
        """
        Set the random number generator instance.

        Parameters
        ----------
        rng: Optional[Union[Integral, Generator]]
            Random number generator instance.
        """
        self._rng = validate_rng(rng, allow_seed=True)

    def _validate_ar_simulation_params(self, params: dict) -> None:
        """
        Validate the parameters necessary for the simulation.
        """
        required_params = ["resids_lags", "resids_coefs", "resids"]
        for param in required_params:
            if params.get(param) is None:
                # logger.error(f"{param} is not provided.")
                raise ValueError(f"{param} must be provided for the AR model.")

    def _simulate_ar_residuals(
        self,
        lags: np.ndarray,
        coefs: np.ndarray,
        init: np.ndarray,
        max_lag: Integral,
    ) -> np.ndarray:
        """
        Simulates an Autoregressive (AR) process with given lags, coefficients, initial values, and random errors.

        Parameters
        ----------
        lags: np.ndarray
            The lags to be used in the AR process. Can be non-consecutive, but when called from `generate_samples_sieve`, it will be sorted.
        coefs: np.ndarray
            The coefficients corresponding to each lag. Of shape (1, len(lags)). Sorted by `generate_samples_sieve` corresponding to the sorted `lags`.
        init: np.ndarray
            The initial values for the simulation. Should be at least as long as the maximum lag.

        Returns
        -------
        np.ndarray
            The simulated AR process as a 1D NumPy array.

        Raises
        ------
        ValueError
            If `lags` or `coefs` are not provided.
            If `coefs` is not a 1D NumPy array.
            If `coefs` is not the same length as `lags`.
            If `init` is not the same length as `max_lag`.

        TypeError
            If `lags` is not an integer or a list of integers.
        """
        random_errors = self.rng.normal(size=self.n_samples)

        if len(init) < max_lag:
            raise ValueError(
                "Length of 'init' must be at least as long as the maximum lag in 'lags'"
            )

        # In case init is 2d with shape (X, 1), convert it to 1d
        init = init.ravel()
        series = np.zeros(self.n_samples, dtype=init.dtype)
        series[:max_lag] = init

        trend_terms = TSFit._calculate_trend_terms(
            model_type="ar", model=self.fitted_model
        )
        intercepts = self.fitted_model.params[:trend_terms].reshape(
            1, trend_terms
        )

        # Loop through the series, calculating each value based on the lagged values, coefficients, random error, and trend term
        for t in range(max_lag, self.n_samples):
            ar_term = 0
            for i in range(len(lags)):
                ar_term_iter = coefs[0, i] * series[t - lags[i]]
                ar_term += ar_term_iter

            trend_term = 0
            # If the trend is 'c' or 'ct', add a constant term
            if self.fitted_model.model.trend in ["c", "ct"]:
                trend_term += intercepts[0, 0]
            # If the trend is 't' or 'ct', add a linear time trend
            if self.fitted_model.model.trend in ["t", "ct"]:
                trend_term += intercepts[0, -1] * t

            series[t] = ar_term + trend_term + random_errors[t]

        return series

    def simulate_ar_process(
        self,
        resids_lags,
        resids_coefs: np.ndarray,
        resids: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate AR process from the fitted model.

        Parameters
        ----------
        resids_lags: Union[Integral, List[Integral]]
            The lags to be used in the AR process. Can be non-consecutive, but when called from `generate_samples_sieve`, it will be sorted.
        resids_coefs: np.ndarray
            The coefficients corresponding to each lag. Of shape (1, len(lags)). Sorted by `generate_samples_sieve` corresponding to the sorted `lags`.
        resids: np.ndarray
            The initial values for the simulation. Should be at least as long as the maximum lag.

        Returns
        -------
        np.ndarray
            The simulated AR process as a 1D NumPy array.

        Raises
        ------
        ValueError
            If `resids_lags`, `resids_coefs`, or `resids` are not provided.
            If `resids_coefs` is not a 1D NumPy array.
            If `resids_coefs` is not the same length as `resids_lags`.
            If `resids` is not the same length as `X_fitted`.

        TypeError
            If `fitted_model` is not an instance of `AutoRegResultsWrapper`.
            If `resids_lags` is not an integer or a list of integers.
        """
        from statsmodels.tsa.ar_model import AutoRegResultsWrapper

        validate_integers(resids_lags, min_value=1)

        if not isinstance(self.fitted_model, AutoRegResultsWrapper):
            # logger.error("fitted_model must be an instance of AutoRegResultsWrapper.")
            raise TypeError(
                f"fitted_model must be an instance of AutoRegResultsWrapper. Got {type(self.fitted_model)}."
            )

        if self.n_features > 1:
            raise ValueError(
                "Only univariate time series are supported for the AR model."
            )
        if self.n_samples != len(resids):
            raise ValueError(
                "Length of 'resids' must be the same as the number of samples in 'X_fitted'."
            )

        # In case resids is 2d with shape (X, 1), convert it to 1d
        resids = resids.ravel()
        # In case X_fitted is 2d with shape (X, 1), convert it to 1d
        X_fitted = self.X_fitted.ravel()
        # Generate the bootstrap series
        bootstrap_series = np.zeros(self.n_samples, dtype=X_fitted.dtype)
        # Convert resids_lags to a NumPy array if it is not already. When called from `generate_samples_sieve`, it will be sorted.
        resids_lags = (
            np.arange(1, resids_lags + 1)
            if isinstance(resids_lags, Integral)
            else np.array(sorted(resids_lags))
        )  # type: ignore
        # resids_lags.shape: (n_lags,)
        max_lag = np.max(resids_lags)
        if resids_coefs.shape[0] != 1:
            raise ValueError(
                "AR coefficients must be a 1D NumPy array of shape (1, X)"
            )
        if resids_coefs.shape[1] != len(resids_lags):  # type: ignore
            raise ValueError(
                "Length of 'resids_coefs' must be the same as the length of 'lags'"
            )

        # Simulate residuals using the AR model
        simulated_residuals = self._simulate_ar_residuals(
            lags=resids_lags,
            coefs=resids_coefs,
            init=resids[:max_lag],
            max_lag=max_lag,
        )
        # simulated_residuals.shape: (n_samples,)

        bootstrap_series[:max_lag] = X_fitted[:max_lag]

        # Loop through the series, calculating each value based on the lagged values, coefficients, and random error
        for t in range(max_lag, self.n_samples):
            lagged_values = bootstrap_series[t - resids_lags]
            # lagged_values.shape: (n_lags,)
            lagged_values = lagged_values.reshape(-1, 1)
            # lagged_values.shape: (n_lags, 1)
            bootstrap_series[t] = (
                resids_coefs @ lagged_values + simulated_residuals[t]
            )

        return bootstrap_series.reshape(-1, 1)

    def _simulate_non_ar_residuals(self) -> np.ndarray:
        """
        Simulate residuals according to the model type.

        Returns
        -------
        np.ndarray
            The simulated residuals.
        """
        from arch.univariate.base import ARCHModelResult
        from statsmodels.tsa.arima.model import ARIMAResultsWrapper
        from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
        from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

        rng_seed = (
            self.rng.integers(0, 2**32 - 1)
            if not isinstance(self.rng, Integral)
            else self.rng
        )

        if isinstance(
            self.fitted_model, (ARIMAResultsWrapper, SARIMAXResultsWrapper)
        ):
            return self.fitted_model.simulate(
                nsimulations=self.n_samples + self.burnin,
                random_state=self.rng,
            )
        elif isinstance(self.fitted_model, VARResultsWrapper):
            return self.fitted_model.simulate_var(
                steps=self.n_samples + self.burnin, seed=rng_seed
            )
        elif isinstance(self.fitted_model, ARCHModelResult):
            return self.fitted_model.model.simulate(
                params=self.fitted_model.params,
                nobs=self.n_samples,
                burn=self.burnin,
            )["data"].values
        raise ValueError(f"Unsupported fitted model type {self.fitted_model}.")

    def simulate_non_ar_process(self) -> np.ndarray:
        """
        Simulate a time series from the fitted model.

        Returns
        -------
            np.ndarray: The simulated time series.
        """
        from statsmodels.tsa.arima.model import ARIMAResultsWrapper
        from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
        from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

        simulated_residuals = self._simulate_non_ar_residuals()
        simulated_residuals = np.reshape(
            simulated_residuals, (-1, self.n_features)
        )
        # Discard the burn-in samples for certain models
        if isinstance(
            self.fitted_model,
            (VARResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper),
        ):
            simulated_residuals = simulated_residuals[self.burnin :]
        return self.X_fitted + simulated_residuals

    def generate_samples_sieve(
        self,
        model_type: ModelTypes,
        resids_lags=None,
        resids_coefs: np.ndarray = None,
        resids: np.ndarray = None,
    ) -> np.ndarray:
        """
        Generate a bootstrap sample using the sieve bootstrap.

        Parameters
        ----------
        model_type: ModelTypes
            The model type used for the simulation.
        resids_lags: Optional[Union[Integral, List[Integral]]], optional
            The lags to be used in the AR process. Can be non-consecutive.
        resids_coefs: Optional[np.ndarray], optional
            The coefficients corresponding to each lag. Of shape (1, len(lags)).
        resids: Optional[np.ndarray], optional
            The initial values for the simulation. Should be at least as long as the maximum lag.

        Returns
        -------
        np.ndarray
            The bootstrap sample.

        Raises
        ------
        ValueError
            If `resids_lags`, `resids_coefs`, or `resids` are not provided.
        """
        validate_literal_type(model_type, ModelTypes)
        if model_type == "ar":
            self._validate_ar_simulation_params(
                {
                    "resids_lags": resids_lags,
                    "resids_coefs": resids_coefs,
                    "resids": resids,
                }
            )
            return self.simulate_ar_process(resids_lags, resids_coefs, resids)
        else:
            return self.simulate_non_ar_process()

    def __repr__(self) -> str:
        return f"TimeSeriesSimulator(fitted_model={self.fitted_model}, n_samples={self.n_samples}, n_features={self.n_features})"

    def __str__(self) -> str:
        return f"TimeSeriesSimulator with {self.n_samples} samples and {self.n_features} features using fitted model {self.fitted_model}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TimeSeriesSimulator):
            return (
                self.fitted_model == other.fitted_model
                and np.array_equal(self.X_fitted, other.X_fitted)
                and self.rng == other.rng
            )
        return False
