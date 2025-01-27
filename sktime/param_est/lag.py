"""Parameter estimators for autoregressive lag order selection.

This module provides functionality for selecting optimal lag orders in
autoregressive models using information criteria.

The main class ARLagOrderSelector implements lag order selection
using various information criteria (AIC, BIC, HQIC) and supports both
sequential and global search strategies.
"""

__author__ = ["satvshr"]
__all__ = ["ARLagOrderSelector"]

from sktime.forecasting.auto_reg import AutoREG
from sktime.param_est.base import BaseParamFitter


class ARLagOrderSelector(BaseParamFitter):
    """Estimate optimal lag order for autoregressive models using information criteria.

    Implements lag order selection for AR models by comparing
    different lag specifications using information criteria.
    Supports both sequential and global search strategies.

    Parameters
    ----------
    maxlag : int
        Maximum number of lags to consider
    ic : str, default="bic"
        Information criterion to use for model selection:
        - "aic" : Akaike Information Criterion
        - "bic" : Bayesian Information Criterion (default)
        - "hqic" : Hannan-Quinn Information Criterion
    glob : bool, default=False
        If True, searches globally across all lag combinations up to maxlag.
        If False, searches sequentially by adding one lag at a time.
    trend : str, default="c"
        Trend to include in the model:
        - "n" : No trend
        - "c" : Constant only
        - "t" : Time trend only
        - "ct" : Constant and time trend
    seasonal : bool, default=False
        Whether to include seasonal dummies in the model
    hold_back : int, optional (default=None)
        Number of initial observations to exclude from the estimation sample
    period : int, optional (default=None)
        Period of the data (used only if seasonal=True)
    missing : str, default="none"
        How to handle missing values:
        - "none" : No handling
        - "drop" : Drop missing observations
        - "raise" : Raise an error

    Attributes
    ----------
    selected_model_ : tuple
        Selected lag order(s) that minimize the information criterion
    ic_value_ : float
        Value of the information criterion for the selected model

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.lag import ARLagOrderSelector
    >>> y = load_airline()
    >>> selector = ARLagOrderSelector(maxlag=12, ic="bic")
    >>> selector.fit(y)
    ARLagOrderSelector(...)
    >>> selector.selected_model_
    (3,)
    >>> selector.ic_value_
    1369.6963340649502

    See Also
    --------
    AutoREG : Autoregressive forecasting model

    Notes
    -----
    The implementation uses OLS estimation and computes information criteria based on
    the likelihood of the AR model. For global search, it evaluates all possible lag
    combinations up to maxlag. For sequential search, it adds one lag at a time until
    maxlag.
    """

    _tags = {
        "X_inner_mtype": "np.ndarray",
        "scitype:X": "Series",
        "capability:missing_values": False,
        "capability:multivariate": False,
        "authors": "satvshr",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        maxlag,
        ic="bic",
        glob=False,
        trend="c",
        seasonal=False,
        hold_back=None,
        period=None,
        missing="none",
    ):
        self.maxlag = maxlag
        self.exog = None
        self.ic = ic
        self.glob = glob
        self.trend = trend
        self.seasonal = seasonal
        self.hold_back = hold_back
        self.period = period
        self.missing = missing
        super().__init__()

    def _fit(self, X):
        """Fit estimator and estimate parameters.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.
            If exog is passed in fit, X is the endogenous variable.

        Returns
        -------
        self : reference to self
        """
        from types import SimpleNamespace

        import numpy as np
        from statsmodels.tsa.ar_model import OLS

        self.model = AutoREG(
            lags=self.maxlag,
            trend=self.trend,
            seasonal=self.seasonal,
            hold_back=self.hold_back,
            period=self.period,
            missing=self.missing,
        )
        self.model.fit(X)
        self.nexog = self.exog.shape[1] if self.exog is not None else 0
        self.y, self.X = self.model._forecaster._y, self.model._forecaster._x
        self.base_col = self.X.shape[1] - self.nexog - self.maxlag
        self.sel = np.ones(self.X.shape[1], dtype=bool)
        self.ics: list[tuple[int | tuple[int, ...], tuple[float, float, float]]] = []

        def _compute_ics(res: SimpleNamespace):
            nobs = res.nobs
            df_model = res.df_model
            sigma2 = 1.0 / nobs * np.sum(res.resid**2)
            llf = -nobs * (np.log(2 * np.pi * sigma2) + 1) / 2
            k = df_model + 1
            aic = -2 * llf + 2 * k
            bic = -2 * llf + np.log(nobs) * k
            hqic = -2 * llf + 2 * k * np.log(np.log(nobs))

            return aic, bic, hqic

        def _ic_no_data():
            mod = SimpleNamespace(
                nobs=self.y.shape[0], endog=self.y, exog=np.empty((self.y.shape[0], 0))
            )
            llf = OLS.loglike(mod, np.empty(0))

            res = SimpleNamespace(
                resid=self.y, nobs=self.y.shape[0], df_model=0, k_constant=0, llf=llf
            )
            return _compute_ics(res)

        if not self.glob:
            self.sel[self.base_col : self.base_col + self.maxlag] = False
            for i in range(self.maxlag + 1):
                self.sel[self.base_col : self.base_col + i] = True
                if not np.any(self.sel):
                    self.ics.append((0, _ic_no_data()))
                    continue
                res = OLS(self.y, self.X[:, self.sel]).fit()
                lags = tuple(j for j in range(1, i + 1))
                lags = 0 if not lags else lags
                self.ics.append((lags, _compute_ics(res)))
        else:
            bits = np.arange(2**self.maxlag, dtype=np.int32)[:, None]
            bits = bits.view(np.uint8)
            bits = np.unpackbits(bits).reshape(-1, 32)
            for i in range(4):
                bits[:, 8 * i : 8 * (i + 1)] = bits[:, 8 * i : 8 * (i + 1)][:, ::-1]
            masks = bits[:, : self.maxlag]
            for mask in masks:
                self.sel[self.base_col : self.base_col + self.maxlag] = mask
                if not np.any(self.sel):
                    self.ics.append((0, _ic_no_data()))
                    continue
                res = OLS(self.y, self.X[:, self.sel]).fit()
                lags = tuple(np.where(mask)[0] + 1)
                lags = 0 if not lags else lags
                self.ics.append((lags, _compute_ics(res)))

        key_loc = {"aic": 0, "bic": 1, "hqic": 2}[self.ic]
        self.ics = sorted(self.ics, key=lambda x: x[1][key_loc])
        self.selected_model_ = self.ics[0][0]
        self.ic_value_ = self.ics[0][1][key_loc]

        return self

    def get_fitted_params(self):
        """Get fitted parameters."""
        return {
            "selected_lags": self.selected_model_,
            "ic_value": self.ic_value_,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class
        """
        params1 = {"maxlag": 5}
        params2 = {
            "maxlag": 12,
            "ic": "aic",
            "seasonal": True,
            "period": 4,
        }

        return [params1, params2]
