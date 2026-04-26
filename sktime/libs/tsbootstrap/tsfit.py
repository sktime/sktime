from __future__ import annotations

import math
import warnings
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted

from tsbootstrap.ranklags import RankLags
from tsbootstrap.time_series_model import TimeSeriesModel
from tsbootstrap.utils.types import (
    ModelTypes,
    OrderTypes,
    OrderTypesWithoutNone,
)
from tsbootstrap.utils.validate import (
    validate_literal_type,
    validate_X,
    validate_X_and_y,
)


class TSFit(BaseEstimator, RegressorMixin):
    """
    Performs fitting for various time series models including 'ar', 'arima', 'sarima', 'var', and 'arch'.

    Attributes
    ----------
    rescale_factors : dict
        Rescaling factors for the input data and exogenous variables.
    model : Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]
        The fitted model.

    Methods
    -------
    fit(X, y=None)
        Fit the chosen model to the data.
    get_coefs()
        Return the coefficients of the fitted model.
    get_intercepts()
        Return the intercepts of the fitted model.
    get_residuals()
        Return the residuals of the fitted model.
    get_fitted_X()
        Return the fitted values of the model.
    get_order()
        Return the order of the fitted model.
    predict(X, n_steps=1)
        Predict future values using the fitted model.
    score(X, y_true)
        Compute the R-squared score for the fitted model.

    Raises
    ------
    ValueError
        If the model type or the model order is invalid.

    Notes
    -----
    The following table shows the valid model types and their corresponding orders.

    +--------+-------------------+-------------------+
    | Model  | Valid orders      | Invalid orders    |
    +========+===================+===================+
    | 'ar'   | int               | list, tuple       |
    +--------+-------------------+-------------------+
    | 'arima'| tuple of length 3 | int, list, tuple  |
    +--------+-------------------+-------------------+
    | 'sarima'| tuple of length 4| int, list, tuple  |
    +--------+-------------------+-------------------+
    | 'var'  | int               | list, tuple       |
    +--------+-------------------+-------------------+
    | 'arch' | int               | list, tuple       |
    +--------+-------------------+-------------------+

    Examples
    --------
    >>> from tsbootstrap import TSFit
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 1))
    >>> fit_obj = TSFit(order=2, model_type='ar')  # doctest: +SKIP
    >>> fit_obj.fit(X)  # doctest: +SKIP
    TSFit(order=2, model_type='ar')
    >>> fit_obj.get_coefs()  # doctest: +SKIP
    array([[ 0.003, -0.002]])
    >>> fit_obj.get_intercepts()  # doctest: +SKIP
    array([0.001])
    >>> fit_obj.get_residuals()  # doctest: +SKIP
    array([[ 0.001],
              [-0.002],
                [-0.002],
                    [-0.002],
                        [-0.002], ...
    >>> fit_obj.get_fitted_X()  # doctest: +SKIP
    array([[ 0.001],
                [-0.002],
                    [-0.002],
                        [-0.002],
                            [-0.002], ...
    >>> fit_obj.get_order()  # doctest: +SKIP
    2
    >>> fit_obj.predict(X, n_steps=5)  # doctest: +SKIP
    array([[ 0.001],
                [-0.002],
                    [-0.002],
                        [-0.002],
                            [-0.002], ...
    >>> fit_obj.score(X, X)  # doctest: +SKIP
    0.999
    """

    _tags = {"python_dependencies": ["arch", "statsmodels"]}

    def __init__(
        self, order: OrderTypesWithoutNone, model_type: ModelTypes, **kwargs
    ) -> None:
        """
        Initialize the TSFit object.

        Parameters
        ----------
        order : int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]
            Order of the model.
        model_type : str
            Type of the model.
        **kwargs
            Additional parameters to be passed to the model.

        Raises
        ------
        ValueError
            If the model type or the model order is invalid.
        """
        self.model_type = model_type
        self.order = order
        self.rescale_factors = {}
        self.model_params = kwargs

    @property
    def model_type(self) -> str:
        """The type of the model."""
        return self._model_type

    @model_type.setter
    def model_type(self, value: ModelTypes) -> None:
        """Set the model type."""
        validate_literal_type(value, ModelTypes)
        value = value.lower()
        self._model_type = value

    @property
    def order(self) -> OrderTypesWithoutNone:
        """The order of the model."""
        return self._order

    @order.setter
    def order(self, value) -> None:
        """Set the order of the model."""
        if not isinstance(value, (Integral, list, tuple)):  # noqa: UP038
            raise TypeError(
                f"Invalid order '{value}', should be an integer, list, or tuple."
            )

        if isinstance(value, list) and len(value) > 1:
            value_orig = value
            value = sorted(value)
            if value != value_orig:
                warning_msg = f"Order '{value_orig}' is a list. Sorting the list to '{value}'."
                warnings.warn(warning_msg, stacklevel=2)

        if isinstance(value, (list, tuple)) and len(value) == 0:  # noqa: UP038
            raise ValueError(
                f"Invalid order '{value}', should be a non-empty list/tuple."
            )

        if isinstance(value, tuple) and self.model_type not in [
            "arima",
            "sarima",
        ]:
            raise ValueError(
                f"Invalid order '{value}', should be an integer for model type '{self.model_type}'"
            )

        if isinstance(value, Integral) and self.model_type in {
            "sarima",
            "arima",
        }:
            if self.model_type == "sarima":
                value = (value, 0, 0, value + 1)
                warning_msg = f"{self.model_type.upper()} model requires a tuple of order (p, d, q, s), where d is the order of differencing and s is the seasonal period. Setting d=0, q=0, and s=2."
            else:
                value = (value, 0, 0)
                warning_msg = f"{self.model_type.upper()} model requires a tuple of order (p, d, q), where d is the order of differencing. Setting d=0, q=0."
            warnings.warn(warning_msg, stacklevel=2)

        self._order = value

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            When set to True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return {
            "order": self.order,
            "model_type": self.model_type,
            **self.model_params,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Estimator parameters.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.model_params[key] = value
        return self

    def __repr__(self):
        """
        Official string representation of a TSFit object.
        """
        return f"TSFit(order={self.order}, model_type='{self.model_type}')"

    def fit(self, X: np.ndarray, y=None) -> TSFit:
        """
        Fit the chosen model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        y : np.ndarray, optional
            Exogenous variables, by default None.

        Returns
        -------
        TSFit
            The fitted TSFit object.

        Raises
        ------
        ValueError
            If the model type or the model order is invalid.
        RuntimeError
            If the maximum number of iterations is reached before the variance is within the desired range.
        """
        # Check if the input shapes are valid
        X, y = validate_X_and_y(
            X,
            y,
            model_is_var=self.model_type == "var",
            model_is_arch=self.model_type == "arch",
        )

        def _rescale_inputs(X: np.ndarray, y=None):
            """
            Rescale the inputs to ensure that the variance of the input data is within the interval [1, 1000].

            Parameters
            ----------
            X : np.ndarray
                The input data.
            y : np.ndarray, optional
                The exogenous variables, by default None.

            Returns
            -------
            Tuple[np.ndarray, np.ndarray, Tuple[float, List[float] | None]]
                A tuple containing the rescaled input data, the rescaled exogenous variables, and the rescaling factors used.

            Raises
            ------
            RuntimeError
                If the maximum number of iterations is reached before the variance is within the desired range.
            """

            def rescale_array(arr: np.ndarray, max_iter: int = 100):
                """
                Iteratively rescales an array to ensure its variance is within the interval [1, 1000].

                Parameters
                ----------
                arr : np.ndarray
                    The input array to be rescaled.
                max_iter : int, optional
                    The maximum number of iterations for rescaling, by default 100.

                Returns
                -------
                Tuple[np.ndarray, float]
                    A tuple containing the rescaled array and the total rescaling factor used.

                Raises
                ------
                RuntimeError
                    If the maximum number of iterations is reached before the variance is within the desired range.
                """
                variance = np.var(arr)
                if math.isclose(variance, 0, abs_tol=0.01):
                    raise RuntimeError(
                        "Variance of the input data is 0. Cannot rescale the input data."
                    )
                total_rescale_factor = 1
                iterations = 0

                while not 1 <= variance <= 1000:
                    if iterations >= max_iter:
                        raise RuntimeError(
                            f"Maximum iterations ({max_iter}) reached. Variance is still not in the range [1, 1000]. Variance = {variance}. Results from the ARCH/GARCH model can not be trusted."
                        )

                    rescale_factor = np.sqrt(100 / variance)
                    arr = arr * rescale_factor
                    total_rescale_factor *= rescale_factor
                    variance = np.var(arr)
                    iterations += 1

                return arr, total_rescale_factor

            X, x_rescale_factor = rescale_array(X)

            if y is not None:
                y_rescale_factors = []
                for i in range(y.shape[1]):
                    y[:, i], factor = rescale_array(y[:, i])
                    y_rescale_factors.append(factor)
            else:
                y_rescale_factors = None

            return X, y, (x_rescale_factor, y_rescale_factors)

        fit_func = TimeSeriesModel(X=X, y=y, model_type=self.model_type)
        self.model = fit_func.fit(order=self.order, **self.model_params)
        if self.model_type == "arch":
            (
                X,
                y,
                (x_rescale_factor, y_rescale_factors),
            ) = _rescale_inputs(X, y)
            self.rescale_factors["x"] = x_rescale_factor
            self.rescale_factors["y"] = y_rescale_factors

        return self

    def get_coefs(self) -> np.ndarray:
        """
        Return the coefficients of the fitted model.

        Returns
        -------
        np.ndarray
            The coefficients of the fitted model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the coefficients depends on the model type.

        +--------+---------------------------------+
        | Model  | Coefficient shape               |
        +========+=================================+
        | 'ar'   | (1, order)                      |
        +--------+---------------------------------+
        | 'arima'| (1, order)                      |
        +--------+---------------------------------+
        | 'sarima'| (1, order)                     |
        +--------+---------------------------------+
        | 'var'  | (n_features, n_features, order) |
        +--------+---------------------------------+
        | 'arch' | (1, order)                      |
        +--------+---------------------------------+
        """
        # Check if the model is already fitted
        check_is_fitted(self, ["model"])

        if self.model_type != "arch":
            n_features = (
                self.model.model.endog.shape[1]
                if len(self.model.model.endog.shape) > 1
                else 1
            )
        else:
            n_features = (
                self.model.model.y.shape[1]
                if len(self.model.model.y.shape) > 1
                else 1
            )
        return self._get_coefs_helper(self.model, n_features)

    def get_intercepts(self) -> np.ndarray:
        """
        Return the intercepts of the fitted model.

        Returns
        -------
        np.ndarray
            The intercepts of the fitted model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the intercepts depends on the model type.

        +--------+---------------------------+
        | Model  | Intercept shape           |
        +========+===========================+
        | 'ar'   | (1, trend_terms)          |
        +--------+---------------------------+
        | 'arima'| (1, trend_terms)          |
        +--------+---------------------------+
        | 'sarima'| (1, trend_terms)         |
        +--------+---------------------------+
        | 'var'  | (n_features, trend_terms) |
        +--------+---------------------------+
        | 'arch' | (0,)                      |
        +--------+---------------------------+
        """
        # Check if the model is already fitted
        check_is_fitted(self, ["model"])
        n_features = (
            self.model.model.endog.shape[1]
            if len(self.model.model.endog.shape) > 1
            else 1
        )
        return self._get_intercepts_helper(self.model, n_features)

    def get_residuals(self) -> np.ndarray:
        """
        Return the residuals of the fitted model.

        Returns
        -------
        np.ndarray
            The residuals of the fitted model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the residuals depends on the model type.

        +--------+-------------------+
        | Model  | Residual shape    |
        +========+===================+
        | 'ar'   | (n, 1)            |
        +--------+-------------------+
        | 'arima'| (n, 1)            |
        +--------+-------------------+
        | 'sarima'| (n, 1)           |
        +--------+-------------------+
        | 'var'  | (n, k)            |
        +--------+-------------------+
        | 'arch' | (n, 1)            |
        +--------+-------------------+
        """
        # Check if the model is already fitted
        check_is_fitted(self, ["model"])
        return self._get_residuals_helper(self.model)

    def get_fitted_X(self) -> np.ndarray:
        """
        Return the fitted values of the model.

        Returns
        -------
        np.ndarray
            The fitted values of the model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the fitted values depends on the model type.

        +--------+--------------------+
        | Model  | Fitted values shape|
        +========+====================+
        | 'ar'   | (n, 1)             |
        +--------+--------------------+
        | 'arima'| (n, 1)             |
        +--------+--------------------+
        | 'sarima'| (n, 1)            |
        +--------+--------------------+
        | 'var'  | (n, k)             |
        +--------+--------------------+
        | 'arch' | (n, 1)             |
        +--------+--------------------+
        """
        # Check if the model is already fitted
        check_is_fitted(self, ["model"])
        return self._get_fitted_X_helper(self.model)

    def get_order(self) -> OrderTypesWithoutNone:
        """
        Return the order of the fitted model.

        Returns
        -------
        OrderTypesWithoutNone
            The order of the fitted model.

        Raises
        ------
        NotFittedError
            If the model is not fitted.

        Notes
        -----
        The shape of the order depends on the model type.

        +--------+-------------------+
        | Model  | Order shape       |
        +========+===================+
        | 'ar'   | int               |
        +--------+-------------------+
        | 'arima'| tuple of length 3 |
        +--------+-------------------+
        | 'sarima'| tuple of length 4|
        +--------+-------------------+
        | 'var'  | int               |
        +--------+-------------------+
        | 'arch' | int               |
        +--------+-------------------+
        """
        # Check if the model is already fitted
        check_is_fitted(self, ["model"])
        return self._get_order_helper(self.model)

    def predict(self, X: np.ndarray, y=None, n_steps: int = 1) -> np.ndarray:
        """
        Predict time series values using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        y : np.ndarray, optional
            Exogenous variables, by default None.
        n_steps : int, optional
            Number of steps to forecast, by default 1.

        Returns
        -------
        np.ndarray
            Predicted values.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        # Check if the model is already fitted
        check_is_fitted(self, ["model"])
        # Check if the input shapes are valid
        X = validate_X(X, model_is_var=self.model_type == "var")
        if self.model_type == "var":
            return self.model.forecast(X, n_steps, exog_future=y)
        elif self.model_type == "arch":
            # Adjust the code according to how ARCH predictions are made in your setup
            return (
                self.model.forecast(horizon=n_steps, x=y, method="analytic")
                .mean.values[-1]
                .ravel()
            )
        elif self.model_type in ["ar", "arima", "sarima"]:
            # For AutoReg, ARIMA, and SARIMA, use the built-in forecast method
            return self.model.forecast(steps=n_steps, exog=y)

    def score(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the R-squared score for the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y_true : np.ndarray
            The true values.

        Returns
        -------
        float
            The R-squared score.

        Raises
        ------
        NotFittedError
            If the model is not fitted.
        ValueError
            If the number of lags is greater than the length of the input data.
        """
        y_pred = self.predict(X)
        # Use r2 as the score
        return r2_score(y_true, y_pred)

    # These helper methods are internal and still take the model as a parameter.
    # They can be used by the public methods above which do not take the model parameter.

    def _get_coefs_helper(self, model, n_features) -> np.ndarray:
        trend_terms = TSFit._calculate_trend_terms(self.model_type, model)
        if self.model_type == "var":
            # Exclude the trend terms and reshape the remaining coefficients
            return (
                model.params[trend_terms:]
                .reshape(n_features, self.get_order(), n_features)
                .transpose(0, 2, 1)
            )
        # shape = (n_features, order, n_features)

        elif self.model_type == "ar":
            # Exclude the trend terms
            if isinstance(self.order, list):
                # Autoreg does not sort the passed lags, but the output from model.params is sorted
                coefs = np.zeros((1, len(self.order)))
                for i, _ in enumerate(self.order):
                    # Exclude the trend terms
                    coefs[0, i] = model.params[trend_terms + i]
            else:
                # Exclude the trend terms
                coefs = model.params[trend_terms:].reshape(1, -1)
            # shape = (1, order)
            return coefs

        elif self.model_type in ["arima", "sarima"]:
            # Exclude the trend terms
            # shape = (1, order)
            return model.params[trend_terms:].reshape(1, -1)

        elif self.model_type == "arch":
            # ARCH models don't include trend terms by default, so just return the params as is
            return model.params

    def _get_intercepts_helper(self, model, n_features) -> np.ndarray:
        trend_terms = TSFit._calculate_trend_terms(self.model_type, model)
        if self.model_type == "var":
            # Include just the trend terms and reshape
            return model.params[:trend_terms].reshape(n_features, trend_terms)
            # shape = (n_features, trend_terms)
        elif self.model_type in ["ar", "arima", "sarima"]:
            # Include just the trend terms
            return model.params[:trend_terms].reshape(1, trend_terms)
            # shape = (1, trend_terms)
        elif self.model_type == "arch":
            # ARCH models don't include trend terms by default, so just return the params as is
            return np.array([])

    @staticmethod
    def _calculate_trend_terms(model_type: str, model) -> int:
        """
        Determine the number of trend terms based on the 'trend' attribute of the model.
        """
        if model_type in ["ar", "arima", "sarima"]:
            trend = model.model.trend
            if trend == "n" or trend is None:
                trend_terms = 0
            elif trend in ["c", "t"]:
                trend_terms = 1
            elif trend == "ct":
                trend_terms = 2
            else:
                raise ValueError(f"Unknown trend term: {trend}")
            return trend_terms

        elif model_type == "var":
            trend = model.trend
            if trend == "nc":
                trend_terms_per_variable = 0
            elif trend == "c":
                trend_terms_per_variable = 1
            elif trend == "ct":
                trend_terms_per_variable = 2
            elif trend == "ctt":
                trend_terms_per_variable = 3
            else:
                raise ValueError(f"Unknown trend term: {trend}")
            return trend_terms_per_variable

    def _get_residuals_helper(self, model) -> np.ndarray:
        model_resid = model.resid

        # Ensure model_resid has the correct shape, (n, 1) or (n, k)
        if model_resid.ndim == 1:
            model_resid = model_resid.reshape(-1, 1)

        if self.model_type in ["ar", "var"]:
            max_lag = (
                self.model.model.endog.shape[0] - model_resid.shape[0]
            )  # np.max(self.get_order())
            values_to_add_back = self.model.model.endog[:max_lag]

            # Ensure values_to_add_back has the same shape as model_resid
            if values_to_add_back.ndim != model_resid.ndim:
                values_to_add_back = values_to_add_back.reshape(-1, 1)

            model_resid = np.vstack((values_to_add_back, model_resid))

        if self.model_type == "arch":
            model_resid = model_resid / self.rescale_factors["x"]

        return model_resid

    def _get_fitted_X_helper(self, model) -> np.ndarray:
        if self.model_type != "arch":
            model_fittedvalues = model.fittedvalues

            # Ensure model_fittedvalues has the correct shape, (n, 1) or (n, k)
            if model_fittedvalues.ndim == 1:
                model_fittedvalues = model_fittedvalues.reshape(-1, 1)

            if self.model_type in ["ar", "var"]:
                max_lag = (
                    self.model.model.endog.shape[0]
                    - model_fittedvalues.shape[0]
                )  # np.max(self.get_order())
                values_to_add_back = self.model.model.endog[:max_lag]

                # Ensure values_to_add_back has the same shape as model_fittedvalues
                if values_to_add_back.ndim != model_fittedvalues.ndim:
                    values_to_add_back = values_to_add_back.reshape(-1, 1)

                model_fittedvalues = np.vstack(
                    (values_to_add_back, model_fittedvalues)
                )
            return model_fittedvalues

        else:
            return (
                model.resid + model.conditional_volatility
            ) / self.rescale_factors["x"]

    def _get_order_helper(self, model) -> OrderTypesWithoutNone:
        """
        Return the order of the fitted model.
        """
        if self.model_type == "arch":
            return model.model.volatility.p
        elif self.model == "var":
            return model.k_ar
        elif self.model_type == "ar" and isinstance(self.order, list):
            return sorted(self.order)
        else:
            return self.order

    def _lag(self, X: np.ndarray, n_lags: int) -> np.ndarray:
        """
        Lag the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        n_lags : int
            The number of lags.

        Returns
        -------
        np.ndarray
            The lagged data.

        Raises
        ------
        ValueError
            If the number of lags is greater than the length of the input data.
        """
        if len(X) < n_lags:
            raise ValueError(
                "Number of lags is greater than the length of the input data."
            )
        return np.column_stack(
            [X[i : -(n_lags - i), :] for i in range(n_lags)]
        )


class TSFitBestLag(BaseEstimator, RegressorMixin):
    """
    A class used to fit time series data and find the best lag for forecasting.

    Attributes
    ----------
    rank_lagger : RankLags
        An instance of the RankLags class.
    ts_fit : TSFit
        An instance of the TSFit class.
    model : Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]
        The fitted time series model.
    rescale_factors : Dict[str, Union[float, List[float] | None]]
        The rescaling factors used for the input data and exogenous variables.

    Methods
    -------
    fit(X, y=None)
        Fit the time series model to the data.
    get_coefs()
        Return the coefficients of the fitted model.
    get_intercepts()
        Return the intercepts of the fitted model.
    get_residuals()
        Return the residuals of the fitted model.
    get_fitted_X()
        Return the fitted values of the model.
    get_order()
        Return the order of the fitted model.
    get_model()
        Return the fitted time series model.
    predict(X, n_steps=1)
        Predict future values using the fitted model.
    score(X, y_true)
        Compute the R-squared score for the fitted model.
    """

    def __init__(
        self,
        model_type: str,
        max_lag: int = 10,
        order: OrderTypes = None,
        save_models=False,
        **kwargs,
    ):
        self.model_type = model_type
        self.max_lag = max_lag
        self.order = order
        self.save_models = save_models
        self.model_params = kwargs
        self.rank_lagger = None
        self.ts_fit = None
        self.model = None
        self.rescale_factors = {"x": 1, "y": None}

    def _compute_best_order(self, X) -> int:
        """
        Internal method to compute the best order for the given data.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        int
            The best order for the given data.
        """
        self.rank_lagger = RankLags(
            X=X,
            max_lag=self.max_lag,
            model_type=self.model_type,
            save_models=self.save_models,
        )
        best_order = self.rank_lagger.estimate_conservative_lag()
        return best_order

    def fit(self, X: np.ndarray, y=None):
        """
        Fit the time series model to the data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : np.ndarray, optional, default=None
            Exogenous variables to include in the model.

        Returns
        -------
        self
            The fitted model.
        """
        if self.order is None:
            self.order = self._compute_best_order(X)
            if self.save_models:
                self.model = self.rank_lagger.get_model(self.order)
        self.ts_fit = TSFit(
            order=self.order, model_type=self.model_type, **self.model_params
        )
        self.model = self.ts_fit.fit(X, y=y).model
        self.rescale_factors = self.ts_fit.rescale_factors
        return self

    def get_coefs(self) -> np.ndarray:
        """
        Return the coefficients of the fitted model.

        Returns
        -------
        np.ndarray
            The coefficients of the fitted model.
        """
        return self.ts_fit.get_coefs()

    def get_residuals(self) -> np.ndarray:
        """
        Return the residuals of the fitted model.

        Returns
        -------
        np.ndarray
            The residuals of the fitted model.
        """
        return self.ts_fit.get_residuals()

    def get_fitted_X(self) -> np.ndarray:
        """
        Return the fitted values of the model.

        Returns
        -------
        np.ndarray
            The fitted values of the model.
        """
        return self.ts_fit.get_fitted_X()

    def get_order(self) -> OrderTypesWithoutNone:
        """
        Return the order of the fitted model.

        Returns
        -------
        int, List[int], Tuple[int, int, int], Tuple[int, int, int, int]
            The order of the fitted model.
        """
        return self.ts_fit.get_order()

    def get_model(self):
        """
        Return the fitted time series model.

        Returns
        -------
        Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]
            The fitted time series model.

        Raises
        ------
        ValueError
            If models were not saved during initialization.
        """
        if self.save_models:
            return self.rank_lagger.get_model(self.order)
        else:
            raise ValueError(
                "Models were not saved. Please set save_models=True during initialization."
            )

    def predict(self, X: np.ndarray, n_steps: int = 1):
        """
        Predict future values using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        n_steps : int, optional, default=1
            The number of steps to predict.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        return self.ts_fit.predict(X, n_steps)

    def score(self, X: np.ndarray, y_true: np.ndarray):
        """
        Compute the R-squared score for the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y_true : np.ndarray
            The true values of the target variable.

        Returns
        -------
        float
            The R-squared score.
        """
        return self.ts_fit.score(X, y_true)

    def __repr__(self) -> str:
        return f"TSFit(model_type={self.model_type}, order={self.order}, model_params={self.model_params})"

    def __str__(self) -> str:
        return f"TSFit using model_type={self.model_type} with order={self.order} and additional parameters {self.model_params}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TSFit):
            return (
                self.model_type == other.model_type
                and self.order == other.order
                and self.rescale_factors == other.rescale_factors
                and self.model == other.model
                and self.model_params == other.model_params
            )
        return False
