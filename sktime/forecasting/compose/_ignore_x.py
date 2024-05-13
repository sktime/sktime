# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements compositor for ignoring exogenous data."""

__author__ = ["fkiraly"]

from sktime.forecasting.base._delegate import _DelegatedForecaster


class IgnoreX(_DelegatedForecaster):
    """Compositor for ignoring exogenous variables.

    Composing with IgnoreX instructs the wrapped forecaster to ignore exogenous
    data. This is useful for testing the impact of exogenous data on forecasts,
    or for use in tuning hyperparameters of the forecaster.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster descendant instance
        The forecaster to wrap.
    ignore_x : bool, optional (default=True)
        Whether to ignore exogenous data or not, this parameter is useful for tuning.
        True: ignore exogenous data, X is not passed on to ``forecaster``
        False: use exogenous data, X is passed on to ``forecaster``

    Attributes
    ----------
    forecaster_ : clone of forecaster
        The fitted forecaster.
    """

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "forecaster_"

    _tags = {
        "ignores-exogeneous-X": True,
    }

    def __init__(self, forecaster, ignore_x=True):
        self.forecaster = forecaster
        self.ignore_x = ignore_x

        super().__init__()

        self.forecaster_ = forecaster.clone()

        self._set_delegated_tags(self.forecaster_)
        self.set_tags(**{"ignores-exogeneous-X": True})

        if not ignore_x:
            self.set_tags(**{"ignores-exogeneous-X": ignore_x})

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
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster

        f = NaiveForecaster()

        params1 = {"forecaster": f}
        params2 = {"forecaster": f, "ignore_x": False}

        return [params1, params2]
