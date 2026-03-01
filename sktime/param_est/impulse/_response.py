"""Parameter estimation for univariate/multivariate impulse response function."""

author = ["OldPatrick"]
all = ["ImpulseResponseFunction"]

import numpy as np
import pandas as pd
import warnings

from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.api import VECM
from statsmodels.tsa.statespace.varmax import VARMAX

from sktime.param_est.base import BaseParamFitter

MODEL_MAPPING = {
    "VARMAX": VARMAX,
    "VECM": VECM,
    "DynamicFactor": DynamicFactor,
}


class ImpulseResponseFunction(BaseParamFitter):
    """Calculation of Impulse Response Parameters for various time-series forecasters.

    Direct interface to 
    ``statsmodels.tsa.statespace.[any_non_var_vecm_model].[MODEL_FROM_MODEL_MAPPING].impulse_responses``
    and
    ``statsmodels.tsa.vector_ar.irf.IRAnalysis``.

    Basically, an impulse reflects a simple input signal into a system. While system itself sounds very vague,
    in the context of time-series such a system can be simply a time series itself or a relationship between
    two time series. Especially in the context of time series, such a relationship is often assumed to be linear 
    and dynamic and therefore to be found in in linear dynamic models such as VAR and VECM, but also in state-space models 
    like Dynamic Factor (ignoring the fact we could write all time-series in statespace forms).

    Going further, an impulse response traces how a one-time shock or sudden change of one time series variable within a 
    system (of several time-series variables) unfolds over time in the whole system of all variables. A common example 
    is how a shock to GDP growth propagates to another country`s GDP growth over time:
    https://www.reed.edu/economics/parker/s14/312/tschapters/S13_Ch_5.pdf (pp. 83-94)

    Parameters
    ----------
    steps : int, optional
        The number of steps for which impulse responses are calculated. 
        Default is 1. Note that for time-invariant models, the initial 
        impulse is not counted as a step, so if steps=1, the output 
        will have 2 entries.

    impulse : int, str or array_like
        If an integer, the state innovation to pulse; must be between 0 and k_posdef-1. 
        If a str, it indicates which column of df the unit (1) impulse is given. 
        Alternatively, a custom impulse vector may be provided; must be shaped 
        k_posdef x 1.

    orthogonalized : bool, optional
        Whether or not to perform impulse using orthogonalized innovations. 
        Note that this will also affect custum impulse vectors. Default is False.

    cumulative : bool, optional
        Whether or not to return cumulative impulse responses. Default is False.

    anchor : int, str, or datetime, optional
        Time point within the sample for the state innovation impulse. 
        Type depends on the index of the given endog in the model. 
        Two special cases are the strings ‘start’ and ‘end’, which refer to 
        setting the impulse at the first and last points of the sample, respectively. 
        Integer values can run from 0 to nobs - 1, or can be negative to apply negative 
        indexing. Finally, if a date/time index was provided to the model, then this 
        argument can be a date string to parse or a datetime type. Default is ‘start’.

    exog : array_like, optional
        New observations of exogenous regressors for our-of-sample periods, if applicable.

    transformed : bool, optional
        Whether or not params is already transformed. Default is True.

    includes_fixed : bool, optional
        If parameters were previously fixed with the fix_params method, this argument 
        describes whether or not params also includes the fixed parameters, in addition 
        to the free parameters. Default is False.


    extend_model=None, missing ...
    extend_kwargs=None, missing ...

    Attributes
    ----------
    irf_ :  np.ndarray 
        Responses for each endogenous variable due to the impulse given by the impulse argument. 
        For a time-invariant model, the impulse responses are given for steps + 1 elements 
        (this gives the “initial impulse” followed by steps responses for the important 
        cases of VAR and SARIMAX models), while for time-varying models the impulse responses are 
        only given for steps elements (to avoid having to unexpectedly provide updated 
        time-varying matrices).

    Examples (rewrite to new rewrote of self._irf)
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.impulse import ImpulseResponseFunction
    >>> from sktime.forecasting.dynamic_factor import DynamicFactor as skdyn
    >>> import pandas as pd
    >>> X = load_airline()
    >>> X2 = X.shift(1).bfill()
    >>> df = pd.DataFrame({"X":X, "X2": X2})
    >>> fitted_model = skdyn(k_factors=1, factor_order=2).fit(df)
    >>> sktime_irf = ImpulseResponseFunction(fitted_model, orthogonalized=True)
    >>> sktime_irf.fit(df)
    ImpulseResponseFunction(...)
    >>> print(sktime_irf_est.get_fitted_params()["irf"])

    Notes
    -----
    Parameter and Attribute description taken from statsmodels.Statsmodels has up to today two different 
    interfaces for impulse responses. The first one is older and seems to serve only VAR, VECM and SVAR models. 
    Within the IRAnalysis class is a plotting option showing directly the fade-out of the impulse response signal. 
    Since an Impulse Response Function measures the change in a dynamic linear relationship, the concept of 
    cointegration plays again a significant role again.

    References
    ----------
    .. [1] Ballarin, G. 2025: Impulse Response Analysis of Structural Nonlinear Time Series Models, 
        https://arxiv.org/html/2305.19089v5

    .. [2] Statsmodels (last visited 15/02/2026):
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.varmax.VARMAX.impulse_responses.html

    .. [3] Statsmodels (last visited 15/02/2026):
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.dynamic_factor.DynamicFactor.impulse_responses.html

    .. [4] Statsmodels (last visited 01/03/2026):
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.irf.IRAnalysis.html
    """

    def test():
        _tags = {
            "X_inner_mtype": "np.ndarray",  # no support of pl.DataFrame
            "capability:missing_values": False,
            "capability:multivariate": True,
            "capability:pairwise": True,
            "authors": "OldPatrick",
            "python_dependencies": "statsmodels",
        }

    def __init__(
        self,
        model=None,  # default fitted None
        steps=1, 
        impulse=0,
        orthogonalized=False,
        cumulative=False,
        anchor=None,
        exog=None, 
        transformed=True, 
        includes_fixed=False,
        extend_model=None, 
        extend_kwargs=None, 
    ):
        self.model = model  # needs a previously fitted model
        self.steps = steps
        self.impulse = impulse
        self.orthogonalized = orthogonalized
        self.cumulative = cumulative
        self.anchor = anchor
        self.exog = exog
        self.transformed = transformed
        self.includes_fixed = includes_fixed

        self.extend_model = extend_model
        self.extend_kwargs = extend_kwargs
         
        super().__init__()

    def _fit(self, X) -> np.ndarray:
        """Fit estimator for univariate and multivariate orthogonal or cumulative irfs.

        Text from statsmodels:
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.impulse_responses.html

        Responses for each endogenous variable due to the impulse given by the
        impulse argument. For a time-invariant model, the impulse responses are
        given for steps + 1 elements (this gives the “initial impulse” followed
        by steps responses for the important cases of VAR and SARIMAX models),
        while for time-varying models the impulse responses are only given for
        steps elements (to avoid having to unexpectedly provide updated
        time-varying matrices). Keep in mind not every model is able to calculate IRF
        for univariate data.

        Parameters
        ----------
        X : array_like, e.g. pd.Series
        Contains the full set of time-series to be investigated, all X AND y.

        Returns
        -------
        self : reference to self
        """
        model_name = self.model.__class__.__name__
        ImportedModel = MODEL_MAPPING[model_name]

        sm_wrapper = self.model._fitted_forecaster

        k_vars = sm_wrapper.model.k_endog
        fitted_params = sm_wrapper.params

        dummy_data = np.zeros((10, k_vars))

        if model_name == "VARMAX":
            p = sm_wrapper.model.k_ar
            q = sm_wrapper.model.k_ma
            trend_type = sm_wrapper.model.trend

            dummy_model = ImportedModel(
                dummy_data, 
                order=(p, q), 
                trend=trend_type
            )
        elif model_name == "DynamicFactor":
            # some models have problem with univariate irf, need warning
            # to show that results can not be calculated univariate, 
            # should not be a Problem for ARIMA for instance.

            if len(X.shape) < 2 or X.shape[1] < 2:
                warnings.warn(
                    f"IRF test: Input requires at least 2 variables."
                    f"Expected shape (n, 2), but got shape {X.shape}."
                    f"Fit your model with at least two variables."
                )

                return None

            k_factors = sm_wrapper.model.k_factors
            factor_order = sm_wrapper.model.factor_order
            error_order = sm_wrapper.model.error_order

            dummy_model = ImportedModel(
                endog=dummy_data,
                k_factors=k_factors,
                factor_order=factor_order,
                error_order=error_order,
                enforce_stationarity=False,
            )
        elif model_name == "VECM":
            pass

        self.irf_ = dummy_model.impulse_responses(
            params=fitted_params, steps=self.steps, orthogonalized=self.orthogonalized
        )

        return self
    
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator/test.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {
            "model": None, 
            "steps": 1, 
            "impulse": 0,
            "orthogonalized": False,
            "cumulative": False,
            "anchor": None,
            "exog": None, 
            "transformed": True, 
            "includes_fixed": False,
            "extend_model":None, 
            "extend_kwargs":None,
        }

        return [params1, params2]