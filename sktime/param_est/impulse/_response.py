"""Parameter estimation for univariate/multivariate impulse response function."""

author = ["PBormann"]
all = ["ImpulseResponseFunction"]

import numpy as np
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.statespace.varmax import VARMAX

from sktime.param_est.base import BaseParamFitter

MODEL_MAPPING = {
    "VARMAX": VARMAX,
    "DynamicFactor": DynamicFactor,
    "DynamicFactorMQ": DynamicFactorMQ,
}


class ImpulseResponseFunction(BaseParamFitter):
    """

    Direct interface to ``statsmodels...``.

    Description

    Parameters
    ----------
    name : type, default=

        *


    Attributes
    ----------
    some :  np.ndarray of float


    Notes
    -----
    Some notes

    See Also
    --------
    ....

    References
    ----------
    .. [1] ...
    .. [2] ...
    """

    def test():
        _tags = {
            "X_inner_mtype": "np.ndarray",  # no support of pl.DataFrame
            "capability:missing_values": False,
            "capability:multivariate": True,
            "capability:pairwise": True,
            "authors": "PBormann",
            "python_dependencies": "statsmodels",
        }

    def __init__(
        self,
        model=None,  # default fitted None
    ):
        self.model = model  # needs a previously fitted model

        super().__init__()

    def fit(self, X) -> np.ndarray:
        """Fit estimator for univariate and multivariate orthogonal or cumulative irfs.

        Text from statsmodels:
        https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.impulse_responses.html

        Responses for each endogenous variable due to the impulse given by the
        impulse argument. For a time-invariant model, the impulse responses are
        given for steps + 1 elements (this gives the “initial impulse” followed
        by steps responses for the important cases of VAR and SARIMAX models),
        while for time-varying models the impulse responses are only given for
        steps elements (to avoid having to unexpectedly provide updated
        time-varying matrices).

        Parameters
        ----------
        X : array_like, e.g. pd.Series
        Contains the full set of time-series to be investigated, all X AND y.

        Returns
        -------
        impulse response : ndarray
        """
        ImportedModel = MODEL_MAPPING[self.model.__class__.__name__]
        proxy_model = ImportedModel(X)
        return proxy_model.fit().impulse_responses()

    def get_irf_from_sktime(self, steps=1, orthogonalized=False):
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

            dummy_model = ImportedModel(dummy_data, order=(p, q), trend=trend_type)
        else:
            k_factors = sm_wrapper.model.k_factors
            factor_order = sm_wrapper.model.factor_order
            error_order = sm_wrapper.model.error_order
            # error_var = sm_wrapper.model.error_var
            # exog = sm_wrapper.model.exog

            dummy_model = ImportedModel(
                endog=dummy_data,
                k_factors=k_factors,
                factor_order=factor_order,
                error_order=error_order,
                enforce_stationarity=False,
            )

        irf = dummy_model.impulse_responses(
            params=fitted_params, steps=steps, orthogonalized=orthogonalized
        )

        return irf
