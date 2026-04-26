from __future__ import annotations

from numbers import Integral

import numpy as np

from tsbootstrap.utils.types import ModelTypes
from tsbootstrap.utils.validate import validate_integers, validate_literal_type


class RankLags:
    """
    A class that uses several metrics to rank lags for time series models.

    Methods
    -------
    rank_lags_by_aic_bic()
        Rank lags based on Akaike information criterion (AIC) and Bayesian information criterion (BIC).
    rank_lags_by_pacf()
        Rank lags based on Partial Autocorrelation Function (PACF) values.
    estimate_conservative_lag()
        Estimate a conservative lag value by considering various metrics.
    get_model(order)
        Retrieve a previously fitted model given an order.

    Examples
    --------
    >>> from tsbootstrap import RankLags
    >>> import numpy as np
    >>> X = np.random.normal(size=(100, 1))
    >>> rank_obj = RankLags(X, model_type='ar')
    >>> rank_obj.estimate_conservative_lag()
    2
    >>> rank_obj.rank_lags_by_aic_bic()
    (array([2, 1]), array([2, 1]))
    >>> rank_obj.rank_lags_by_pacf()
    array([1, 2])
    """

    _tags = {"python_dependencies": "statsmodels"}

    def __init__(
        self,
        X: np.ndarray,
        model_type: ModelTypes,
        max_lag: Integral = 10,
        y=None,
        save_models: bool = False,
    ) -> None:
        """
        Initialize the RankLags object.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        model_type : str
            The type of model to fit. One of 'ar', 'arima', 'sarima', 'var', 'arch'.
        max_lag : int, optional, default=10
            Maximum lag to consider.
        y : np.ndarray, optional, default=None
            Exogenous variables to include in the model.
        save_models : bool, optional, default=False
            Whether to save the models.
        """
        self.X = X
        self.max_lag = max_lag
        self.model_type = model_type
        self.y = y
        self.save_models = save_models
        self.models = []

    @property
    def X(self) -> np.ndarray:
        """
        The input data.

        Returns
        -------
        np.ndarray
            The input data.
        """
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        """
        Set the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("X must be a numpy array.")
        self._X = value

    @property
    def max_lag(self) -> Integral:
        """
        Maximum lag to consider.

        Returns
        -------
        int
            Maximum lag to consider.
        """
        return self._max_lag

    @max_lag.setter
    def max_lag(self, value: Integral) -> None:
        """
        Set the maximum lag to consider.

        Parameters
        ----------
        max_lag : int
            Maximum lag to consider.
        """
        validate_integers(value, min_value=1)
        self._max_lag = value

    @property
    def model_type(self) -> ModelTypes:
        """
        The type of model to fit.

        Returns
        -------
        str
            The type of model to fit.
        """
        return self._model_type

    @model_type.setter
    def model_type(self, value: ModelTypes) -> None:
        """
        Set the type of model to fit.

        Parameters
        ----------
        value : ModelTypes
            The type of model to fit. One of 'ar', 'arima', 'sarima', 'var', 'arch'.
        """
        validate_literal_type(value, ModelTypes)
        self._model_type = value.lower()

    @property
    def y(self) -> np.ndarray:
        """
        Exogenous variables to include in the model.

        Returns
        -------
        np.ndarray
            Exogenous variables to include in the model.
        """
        return self._y

    @y.setter
    def y(self, value: np.ndarray) -> None:
        """
        Set the exogenous variables to include in the model.

        Parameters
        ----------
        y : np.ndarray
            Exogenous variables to include in the model.
        """
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("y must be a numpy array.")
        self._y = value

    def rank_lags_by_aic_bic(self):
        """
        Rank lags based on Akaike information criterion (AIC) and Bayesian information criterion (BIC).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            aic_ranked_lags: Lags ranked by AIC.
            bic_ranked_lags: Lags ranked by BIC.
        """
        from tsbootstrap.tsfit import TSFit

        aic_values = []
        bic_values = []
        for lag in range(1, self.max_lag + 1):
            try:
                fit_obj = TSFit(order=lag, model_type=self.model_type)
                model = fit_obj.fit(X=self.X, y=self.y).model
            except Exception as e:
                # raise RuntimeError(f"An error occurred during fitting: {e}")
                print(f"{e}")
                break
            if self.save_models:
                self.models.append(model)
            aic_values.append(model.aic)
            bic_values.append(model.bic)

        aic_ranked_lags = np.argsort(aic_values) + 1
        bic_ranked_lags = np.argsort(bic_values) + 1

        return aic_ranked_lags, bic_ranked_lags

    def rank_lags_by_pacf(self) -> np.ndarray:
        """
        Rank lags based on Partial Autocorrelation Function (PACF) values.

        Returns
        -------
        np.ndarray
            Lags ranked by PACF values.
        """
        from statsmodels.tsa.stattools import pacf

        # Can only compute partial correlations for lags up to 50% of the sample size. We use the minimum of max_lag and third of the sample size, to allow for other parameters and trends to be included in the model.
        pacf_values = pacf(
            self.X, nlags=max(min(self.max_lag, self.X.shape[0] // 3 - 1), 1)
        )[1:]
        ci = 1.96 / np.sqrt(len(self.X))
        significant_lags = np.where(np.abs(pacf_values) > ci)[0] + 1
        return significant_lags

    def estimate_conservative_lag(self) -> int:
        """
        Estimate a conservative lag value by considering various metrics.

        Returns
        -------
        int
            A conservative lag value.
        """
        aic_ranked_lags, bic_ranked_lags = self.rank_lags_by_aic_bic()
        # PACF is only available for univariate data
        if self.X.shape[1] == 1:
            pacf_ranked_lags = self.rank_lags_by_pacf()
            highest_ranked_lags = set(aic_ranked_lags).intersection(
                bic_ranked_lags, pacf_ranked_lags
            )
        else:
            highest_ranked_lags = set(aic_ranked_lags).intersection(
                bic_ranked_lags
            )

        if not highest_ranked_lags:
            return aic_ranked_lags[-1]
        else:
            return min(highest_ranked_lags)

    def get_model(self, order: int):
        """
        Retrieve a previously fitted model given an order.

        Parameters
        ----------
        order : int
            Order of the model to retrieve.

        Returns
        -------
        Union[AutoRegResultsWrapper, ARIMAResultsWrapper, SARIMAXResultsWrapper, VARResultsWrapper, ARCHModelResult]
            The fitted model.
        """
        return self.models[order - 1] if self.save_models else None
