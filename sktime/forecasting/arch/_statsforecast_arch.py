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

    Direct interface to ``statsforecast.models.GARCH``.

    This implements the Generalized Autoregressive Conditional
    Heteroskedasticity (GARCH) model.

    Constructs a GARCH(p, q) model.

    Parameters
    ----------
    p: int (default 1)
        AR parameter - number of auto-regressive lags.
    q: int (default 1)
        GARCH heteroskedasticity lag parameter - number of lags for variance term.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arch import StatsForecastGARCH
    >>> y = load_airline()
    >>> forecaster = StatsForecastGARCH(p=2,q=1)
    >>> forecaster.fit(y)
    StatsForecastGARCH(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "MMenchero",
            "AzulGarza",
            "jmoralez",
            "eyjo",
            "yarnabrina",
            "arnaujc91",
            "luca-miniati",
        ],
        # MMenchero, AzulGarza, and jmoralez for statsforecast GARCH
        "maintainers": ["eyjo"],
        # estimator type
        # --------------
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["statsforecast>=1.5.0"],
    }

    def __init__(
        self,
        p=1,
        q=1,
    ):
        self.p = p
        self.q = q

        super().__init__()

    def _get_statsforecast_class(self):
        from statsforecast.models import GARCH as _GARCH

        return _GARCH

    def _get_statsforecast_params(self):
        return {"p": self.p, "q": self.q}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [{}, {"p": 1, "q": 1}]
        return params


class StatsForecastARCH(_GeneralisedStatsForecastAdapter):
    """StatsForecast ARCH estimator.

    Direct interface to ``statsforecast.models.ARCH``.

    This implements the Autoregressive Conditional
    Heteroskedasticity (ARCH) model.

    Constructs an ARCH(p) model.

    Parameters
    ----------
    p: int (default 1)
        AR parameter - number of auto-regressive lags.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arch import StatsForecastARCH
    >>> y = load_airline()
    >>> forecaster = StatsForecastARCH(p=2)
    >>> forecaster.fit(y)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["eyjo"],
        "maintainers": ["eyjo"],
        # estimator type
        # --------------
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["statsforecast>=1.5.0"],
    }

    def __init__(
        self,
        p=1,
    ):
        self.p = p

        super().__init__()

    def _get_statsforecast_class(self):
        from statsforecast.models import ARCH as _ARCH

        return _ARCH

    def _get_statsforecast_params(self):
        return {"p": self.p}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [{}, {"p": 1}]
        return params
