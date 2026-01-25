"""Parameter estimation for univariate/multivariate impulse response function."""

author = ["PBormann"]
all = ["ImpulseResponseFunction"]

from sktime.param_est.base import BaseParamFitter


class ImpulseResponseFunction(BaseParamFitter):
    """Test for cointegration ranks/relationships for VECM Time-Series.

    Direct interface to ``statsmodels...``.

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
    >>> from sktime.param_est.cointegration import JohansenCointegration
    >>> import pandas as pd
    >>> X = load_airline()
    >>> X2 = X.shift(1).bfill()
    >>> df = pd.DataFrame({"X":X, "X2": X2})
    >>> coint_est = JohansenCointegration()
    >>> coint_est.fit(df)
    JohansenCointegration(...)
    >>> print(coint_est.get_fitted_params()["cvm"])
    [[15.0006 17.1481 21.7465]
     [ 2.7055  3.8415  6.6349]]

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
        model=None, #default fitted
        model_name="VARMAX"
    ):
        self.model = model #needs a previously fitted model
        self.model_name = model_name

        super().__init__()

    def fit(self, X):
        if self.model_name == "VARMAX":
            from statsmodels.tsa.statespace.varmax import VARMAX
            proxy_model = VARMAX(X, order=(1, 0))
            return proxy_model.fit().impulse_responses()
        
    def get_irf_from_sktime(self, steps=1):
        import numpy as np
        from statsmodels.tsa.statespace.varmax import VARMAX
        sm_wrapper = self.model._fitted_forecaster
        
        fitted_params = sm_wrapper.params
    
        k_vars = sm_wrapper.model.k_endog
        p = sm_wrapper.model.k_ar
        q = sm_wrapper.model.k_ma
        trend_type = sm_wrapper.model.trend 
        
        # We pass dummy data of zeros, but with the correct shape (k_vars)
        dummy_data = np.zeros((10, k_vars))
        
        dummy_model = VARMAX(
            dummy_data, 
            order=(p, q), 
            trend=trend_type 
        )
        

        irf = dummy_model.impulse_responses(
            params=fitted_params, 
            steps=steps
        )

        return irf