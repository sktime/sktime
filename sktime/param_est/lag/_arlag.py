"""Parameter estimators for autoregressive lag order selection.

This module provides functionality for selecting optimal lag orders in
autoregressive models using information criteria.

The main class ARLagOrderSelector implements lag order selection
using various information criteria (AIC, BIC, HQIC) and supports both
sequential and global search strategies.
"""

__author__ = ["satvshr"]
__all__ = ["ARLagOrderSelector"]

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
    selected_model_ : tuple of int
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
    >>> selector.ic_value_  # doctest: +SKIP
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
        "capability:pairwise": True,
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

    def _fit(self, X, y=None):
        """Fit estimator and estimate parameters.

        Private _fit containing the core logic, called from fit.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Time series to which to fit the estimator.
            If exog is passed in fit, X is the endogenous variable.
        y : array-like, optional (default=None)
            Exogenous variables used for fitting the estimator.
            Acts as the exog parameter in the underlying model.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.ar_model import ar_select_order

        self.results = ar_select_order(
            X,
            exog=y,
            maxlag=self.maxlag,
            ic=self.ic,
            trend=self.trend,
            seasonal=self.seasonal,
            hold_back=self.hold_back,
            period=self.period,
            missing=self.missing,
        )
        self.selected_model_ = self.results.ar_lags
        key_loc = {"aic": 0, "bic": 1, "hqic": 2}[self.ic]
        self.ic_value_ = self.results._ics[0][1][key_loc]

        return self

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
