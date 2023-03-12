# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter estimators for seasonality."""

__author__ = ["fkiraly"]
__all__ = ["PluginParamsForecaster"]

from sktime.forecasting.base._delegate import _DelegatedForecaster


class PluginParamsForecaster(_DelegatedForecaster):
    """Plugs parameters from a parameter estimator into a forecaster.

    In `fit`, first fits `param_est` to data.
    Then, does `forecaster.set_params` with desired/selected parameters.
    After that, behaves as `forecaster` with those parameters set.

    Example: `param_est` seasonality test to determine `sp` parameter;
        `forecaster` being any forecaster with an `sp` parameter.

    Parameters
    ----------
    param_est : parameter estimator, i.e., estimator inheriting from BaseParamFitter
        this is a "blueprint" estimator, state does not change when `fit` is called
    forecaster : sktime forecaster, i.e., estimator inheriting from BaseForecaster
        this is a "blueprint" estimator, state does not change when `fit` is called
    params : None, str, list of str, dict with str values/keys, optional, default=None
        determines which parameters from param_est are plugged into forecaster and where
        None: all parameters of param_est are plugged into forecaster
            only parameters present in both `forecaster` and `param_est` are plugged in
        list of str: parameters in the list are plugged into parameters of the same name
            only parameters present in both `forecaster` and `param_est` are plugged in
        str: considered as a one-element list of str with the string as single element
        dict: parameter with name of key is plugged into parameter with name of value
            only keys present in `param_est` and values in `forecaster` are plugged in
    update_params : bool, optional, default=False
        whether fitted parameters by param_est_ are to be updated in self.update

    Attributes
    ----------
    param_est_ : sktime parameter estimator, clone of estimator in `param_est`
        this clone is fitted in the pipeline when `fit` is called
    forecaster_ : sktime forecaster, clone of forecaster in `forecaster`
        this clone is fitted in the pipeline when `fit` is called

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.param_est.plugin import PluginParamsForecaster
    >>> from sktime.param_est.seasonality import SeasonalityACF
    >>> from sktime.transformations.series.difference import Differencer
    >>>
    >>> y = load_airline()  # doctest: +SKIP
    >>> sp_est = Differencer() * SeasonalityACF()  # doctest: +SKIP
    >>> fcst = NaiveForecaster()  # doctest: +SKIP
    >>> sp_auto = PluginParamsForecaster(sp_est, fcst)  # doctest: +SKIP
    >>> sp_auto.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    PluginParamsForecaster(...)
    >>> y_pred = sp_auto.predict()  # doctest: +SKIP
    >>> sp_auto.forecaster_.get_params()["sp"]  # doctest: +SKIP
    12

    using dictionary to plug "foo" parameter into "sp"
    >>> from sktime.param_est.fixed import FixedParams
    >>> sp_plugin = PluginParamsForecaster(
    ...     FixedParams({"foo": 12}), NaiveForecaster(), params={"foo": "sp"}
    ... )  # doctest: +SKIP
    """

    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "scitype:y": "both",
        "y_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "fit_is_empty": False,
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods to those of same name in self.forecaster_
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "forecaster_"

    def __init__(self, param_est, forecaster, params=None, update_params=False):
        self.param_est = param_est
        self.param_est_ = param_est.clone()
        self.forecaster = forecaster
        self.forecaster_ = forecaster.clone()
        self.params = params
        self.update_params = update_params

        super(PluginParamsForecaster, self).__init__()
        self.clone_tags(self.forecaster_)
        self.set_tags(**{"fit_is_empty": False})
        # todo: only works for single series now
        #   think about how to deal with vectorization later
        self.set_tags(**{"y_inner_mtype": ["pd.DataFrame", "pd.Series", "np.ndarray"]})

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        # reference to delegate
        forecaster = self._get_delegate()

        # fit the parameter estimator to y
        param_est = self.param_est_
        param_est.fit(y)
        fitted_params = param_est.get_fitted_params()

        # obtain the mapping restricted to param names that are available
        fc_par_names = forecaster.get_params().keys()
        pe_par_names = fitted_params.keys()

        params = self.params
        if params is None:
            param_map = {x: x for x in fitted_params.keys()}
        elif isinstance(params, str):
            param_map = {params: params}
        elif isinstance(params, list):
            param_map = {x: x for x in params}
        elif isinstance(params, dict):
            param_map = params
        else:
            raise TypeError("params must be None, a str, a list of str, or a dict")

        param_map = {x: param_map[x] for x in param_map.keys() if x in fc_par_names}
        param_map = {
            x: param_map[x] for x in param_map.keys() if param_map[x] in pe_par_names
        }
        self.param_map_ = param_map

        # obtain the values of fitted params, and set forecaster to those
        new_params = {x: fitted_params[x] for x in param_map}
        forecaster.set_params(**new_params)

        # fit the forecaster, with the fitted parameter values
        forecaster.fit(y=y, fh=fh, X=X)
        return self

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        # reference to delegate
        forecaster = self._get_delegate()

        # if param_est is not updated, we just update the delegate
        if not self.update_params:
            # note: the inner update_params are different and controlled by the method
            forecaster.update(y=y, X=X, update_params=update_params)
            return self
        # else, we repeat fit on the entire data
        else:
            # fit the parameter estimator to y
            param_est = self.param_est_
            param_est.update(y)
            fitted_params = param_est.get_fitted_params()

            # obtain the values of fitted params, and set forecaster to those
            param_map = self.param_map_
            new_params = {x: fitted_params[x] for x in param_map}
            forecaster.set_params(**new_params)

            # now fit forecaster on entire data
            forecaster.fit(y=self._y, X=self._X)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.param_est.fixed import FixedParams
        from sktime.param_est.seasonality import SeasonalityACF
        from sktime.utils.validation._dependencies import _check_estimator_deps

        # use of dictionary to plug "foo" parameter into "sp", uses mock param_est
        params1 = {
            "forecaster": NaiveForecaster(),
            "param_est": FixedParams({"foo": 12}),
            "params": {"foo": "sp"},
        }
        params = [params1]

        # uses a "real" param est that depends on statsmodels, requires statsmodels
        if _check_estimator_deps(SeasonalityACF, severity="none"):
            # explicit reference to a parameter "sp", present in both estimators
            params2 = {
                "forecaster": NaiveForecaster(),
                "param_est": SeasonalityACF(),
                "params": "sp",
            }
            params = params + [params2]

            # no params given, this should recognize that the intersection is only "sp"
            params3 = {
                "forecaster": NaiveForecaster(),
                "param_est": SeasonalityACF(),
                "update_params": True,
            }
            params = params + [params3]

        return params
