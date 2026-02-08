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
    """

    Direct interface to 
    ``statsmodels.tsa.statespace.[any_non_var_vecm_model].[MODEL_FROM_MODEL_MAPPING].impulse_responses``.

    Description

    Parameters
    ----------
    name : type, default=

        *


    Attributes
    ----------
    some :  np.ndarray of float

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.impulse import ImpulseResponseFunction
    >>> from sktime.forecasting.dynamic_factor import DynamicFactor as dynfc
    >>> import pandas as pd
    >>> X = load_airline()
    >>> X2 = X.shift(1).bfill()
    >>> df = pd.DataFrame({"X":X, "X2": X2})
    >>> fitted_model = dynfc(k_factors=1, factor_order=2).fit(df)
    >>> res = ImpulseResponseFunction(fitted_model).fit(df)

    # missing the fitted params function, need to insert and mabe rework the the irf.fit()

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
            "authors": "OldPatrick",
            "python_dependencies": "statsmodels",
        }

    def __init__(
        self,
        model=None,  # default fitted None
        steps=1, 
        orthogonalized=False
    ):
        self.model = model  # needs a previously fitted model
        self.steps = steps
        self.orthogonalized = orthogonalized

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
        time-varying matrices). Keep in mind not every model is able to calculate IRF
        for univariate data.

        Parameters
        ----------
        X : array_like, e.g. pd.Series
        Contains the full set of time-series to be investigated, all X AND y.

        Returns
        -------
        impulse response : ndarray
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
            # to show that results can not be calculated univariate

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

        irf = dummy_model.impulse_responses(
            params=fitted_params, steps=self.steps, orthogonalized=self.orthogonalized
        )
        return irf