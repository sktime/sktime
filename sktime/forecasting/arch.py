# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements ARCH and GARCH models."""

__author__ = ["eyjo"]

__all__ = [
    "StatsForecastARCH",
    "StatsForecastGARCH",
]


from sktime.forecasting.base.adapters._generalised_statsforecast import (
    _GeneralisedStatsForecastAdapter,
)


class StatsForecastGARCH(_GeneralisedStatsForecastAdapter):
    """StatsForecast GARCH estimator.
    
    This implements the Generalized Autoregressive Conditional 
    Heteroskedasticity (GARCH) model.

    Returns the GARCH model according to p and q parameter input.
    
    Parameters
    ----------
    p: int (default 1)
        Number of lagged versions of the series.
    q: int (default 1)
        Number of moving average for the series.
    alias: str (default "GARCH")
        Custom name of the model. 
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        p = 1,
        q = 1,
    ):
        self.p = p
        self.q = q

        super().__init__()

    def _instantiate_model(self):
        # import inside method to avoid hard dependency
        from statsforecast.models import GARCH as _GARCH

        return _GARCH(
            p=self.p,
            q=self.q,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [{}, {"approximation": True, "p": 1, "q": 1}]
        return params


class StatsForecastARCH(_GeneralisedStatsForecastAdapter):
    """StatsForecast ARCH estimator.
    
    This implements the Autoregressive Conditional 
    Heteroskedasticity (ARCH) model.

    Returns the ARCH model according to p and q parameter input.
    
    Parameters
    ----------
    p: int (default 1)
        Number of lagged versions of the series.
    alias: str (default "GARCH")
        Custom name of the model. 
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        p = 1,
    ):
        self.p = p

        super().__init__()

    def _instantiate_model(self):
        # import inside method to avoid hard dependency
        from statsforecast.models import ARCH as _ARCH

        return _ARCH(
            p=self.p,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [{}, {"approximation": True, "p": 1, "q": 1}]
        return params

