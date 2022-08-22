#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Modular ThetaForecaster."""

__author__ = ["GuzalBulatova"]
__all__ = ["ThetaModularForecaster"]

from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.forecasting.compose._ensemble import _aggregate
from sktime.forecasting.compose._pipeline import TransformedTargetForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.theta import ThetaLinesTransformer


class ThetaModularForecaster(_HeterogenousEnsembleForecaster):
    """Modular theta method for forecasting.

    Modularized implementation of Theta method as defined in [1]_ (TODO: add the
    auto-theta method as described in [2]_).

    Overview: Input :term:`univariate series <Univariate time series>` of length
    "n" and decompose with :class:`ThetaLinesTransformer
    <sktime.transformations.series.theta>` by modifying the local curvature of
    the time series using Theta-coefficient values - `theta` parameter.
    Thansformation gives a pd.DataFrame of shape `len(input series) * len(theta)`.

    The resulting transformed series (Theta-lines) are extrapolated separately.
    The forecasts are then aggregated into one prediction - aunivariate series,
    of `len(fh)`.

    References
    ----------
    .. [1] V.Assimakopoulos et al., "The theta model: a decomposition approach
       to forecasting", International Journal of Forecasting, vol. 16, pp. 521-530,
       2000.
    .. [2] E.Spiliotis et al., "Generalizing the Theta method for
       automatic forecasting ", European Journal of Operational
       Research, vol. 284, pp. 550-558, 2020.


    See Also
    --------
    ThetaLinesTransformer :
        Input series decomposition, a step of the pipeline.

    Notes
    -----
    Existing implementation :
        :class:`sktime.forecasting.theta`.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.theta_new import ThetaModularForecaster
    >>> y = load_airline()
    >>> forecaster = ThetaModularForecaster()
    >>> forecaster.fit(y)
    ThetaModularForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "univariate-only": False,
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_version": ">3.7",
    }

    def __init__(
        self,
        forecasters=None,
        theta_values=(0, 2),
        aggfunc="mean",
        weights=None,
    ):
        super(ThetaModularForecaster, self).__init__(forecasters=forecasters)
        self.forecasters = forecasters
        self.aggfunc = aggfunc
        self.weights = weights
        self.theta_values = theta_values

        forecasters_ = self._check_forecasters(forecasters)

        self._colens = ColumnEnsembleForecaster(forecasters=forecasters_)
        # self.forecasters = self._colens.get_params(deep=True)["forecasters"]

        self.pipe_ = TransformedTargetForecaster(
            steps=[
                ("transformer", ThetaLinesTransformer(theta=self.theta_values)),
                ("forecaster", self._colens),
            ]
        )

    def _check_forecasters(self, forecasters):
        if forecasters is None:
            _forecasters = []
            for i, theta in enumerate(self.theta_values):
                if theta == 0:
                    name = "trend" + str(i)
                    forecaster = (name, PolynomialTrendForecaster(), i)
                else:
                    name = "ses" + str(i)
                    forecaster = (name, ExponentialSmoothing(), i)
                _forecasters.append(forecaster)
        else:
            _forecasters = forecasters
        return _forecasters

    def get_params(self, deep=True):
        """Overwrite parent's method.

        Get `'forecasters'` parameter from ColumnEnsemble in order to comply with
        the implementation of get_params in  _HeterogenousMetaEstimator which
        expects lists of tuples of len 2.
        """
        params = super(ThetaModularForecaster, self).get_params(deep=False)
        del params["forecasters"]
        params.update(self._colens.get_params(deep=deep))
        return params

    def set_params(self, **params):
        """Overwrite parent's method.

        Set `'forecasters'` parameter from ColumnEnsemble in order to comply with
        the implementation of get_params in  _HeterogenousMetaEstimator which
        expects lists of tuples of len 2.
        """
        self.theta_values = params.pop("theta_values")
        self.aggfunc = params.pop("aggfunc")
        self.weights = params.pop("weights")
        colens = ColumnEnsembleForecaster(forecasters=self.forecasters)
        if "forecasters" in params.keys() and params["forecasters"] is None:
            params["forecasters"] = self._check_forecasters(None)
        colens.set_params(**params)
        self._colens = colens
        # self.forecasters = self._colens.get_params(deep=True)["forecasters"]
        self.pipe_ = TransformedTargetForecaster(
            steps=[
                ("transformer", ThetaLinesTransformer(theta=self.theta_values)),
                ("forecaster", self._colens),
            ]
        )
        return self

    # TODO: add forecaster checks

    def _fit(self, y, X=None, fh=None):

        self.pipe_.fit(y=y, X=X, fh=fh)
        return self

    def _predict(self, fh, X=None, return_pred_int=False):
        # Call predict on the forecaster directly, not on the pipeline
        # because of output conversion
        Y_pred = self.pipe_.steps_[-1][-1].predict(fh, X)

        return _aggregate(Y_pred, aggfunc=self.aggfunc, weights=self.weights)

    def _update(self, y, X=None, update_params=True):
        self.pipe_._update(y, X=None, update_params=True)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If
            no special parameters are defined for a value, will return
            `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test
            instance, i.e., `MyClass(**params)` or `MyClass(**params[i])`
            creates a valid test instance. `create_test_instance` uses the first
            (or only) dictionary in `params`.
        """
        # imports
        from sktime.forecasting.naive import NaiveForecaster

        # from sktime.forecasting.trend import PolynomialTrendForecaster

        params0 = {
            "forecasters": [
                ("naive", NaiveForecaster(), 0),
                ("naive1", NaiveForecaster(), 1),
            ]
        }
        # params1 = {"forecasters": NaiveForecaster(),
        #            "theta_values": (-1, 0.5, 42)}
        # params2 = {"forecasters": [PolynomialTrendForecaster(),
        #                              ExponentialSmoothing()]}

        # return [params1, params2]
        return params0
