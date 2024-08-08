# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Plugin composite for substituting parameter estimator fit into forecasters."""

__author__ = ["fkiraly"]
__all__ = ["PluginParamsForecaster"]

from inspect import signature

from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.param_est.plugin._common import _resolve_param_map


class PluginParamsForecaster(_DelegatedForecaster):
    """Plugs parameters from a parameter estimator into a forecaster.

    In ``fit``, first fits ``param_est`` to data passed:

    * ``y`` of ``fit`` is passed as the first arg to ``param_est.fit``
    * ``X`` of ``fit`` is passed as the second arg,
      if ``param_est.fit`` has a second arg
    * ``fh`` of ``fit`` is passed as ``fh``,
      if any remaining arg of ``param_est.fit`` is ``fh``

    Then, does ``forecaster.set_params`` with desired/selected parameters.
    Parameters of the fitted ``param_est`` are passed on to ``forecaster``,
    from/to pairs are as specified by the ``params`` parameter of ``self``, see below.

    Then, fits ``forecaster`` to the data passed in ``fit``.

    After that, behaves identically to ``forecaster`` with those parameters set.
    ``update`` behaviour is controlled by the ``update_params`` parameter.

    Example: ``param_est`` seasonality test to determine ``sp`` parameter;
    ``forecaster`` a forecaster with an ``sp`` parameter,
    e.g., ``ExponentialSmoothing``.

    Parameters
    ----------
    param_est : sktime estimator object with a fit method, inheriting from BaseEstimator
        e.g., estimator inheriting from BaseParamFitter or forecaster
        this is a "blueprint" estimator, state does not change when ``fit`` is called

    forecaster : sktime forecaster, i.e., estimator inheriting from BaseForecaster
        this is a "blueprint" estimator, state does not change when ``fit`` is called

    params : None, str, list of str, dict with str values/keys, optional, default=None
        determines which parameters from ``param_est`` are plugged into forecaster where
        None: all parameters of param_est are plugged into forecaster
        only parameters present in both ``forecaster`` and ``param_est`` are plugged in
        list of str: parameters in the list are plugged into parameters of the same name
        only parameters present in both ``forecaster`` and ``param_est`` are plugged in
        str: considered as a one-element list of str with the string as single element
        dict: parameter with name of value is plugged into parameter with name of key
        only keys present in ``param_est`` and values in ``forecaster`` are plugged in

    update_params : bool, optional, default=False
        whether fitted parameters by param_est_ are to be updated in self.update

    Attributes
    ----------
    param_est_ : sktime parameter estimator, clone of estimator in ``param_est``
        this clone is fitted in the pipeline when ``fit`` is called
    forecaster_ : sktime forecaster, clone of ``forecaster``
        this clone is fitted in the pipeline when ``fit`` is called
    param_map_ : dict
        mapping of parameters from ``param_est_`` to ``forecaster_`` used in ``fit``,
        after filtering for parameters present in both

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.param_est.plugin import PluginParamsForecaster
    >>> from sktime.param_est.seasonality import SeasonalityACF
    >>> from sktime.transformations.series.difference import Differencer
    >>>
    >>> y = load_airline()  # doctest: +SKIP
    >>>
    >>> # sp_est is a seasonality estimator
    >>> # ACF assumes stationarity so we concat with differencing first
    >>> sp_est = Differencer() * SeasonalityACF()  # doctest: +SKIP
    >>>
    >>> # fcst is a forecaster with a "sp" parameter which we want to tune
    >>> fcst = NaiveForecaster()  # doctest: +SKIP
    >>>
    >>> # sp_auto is auto-tuned via PluginParamsForecaster
    >>> sp_auto = PluginParamsForecaster(sp_est, fcst)  # doctest: +SKIP
    >>>
    >>> # fit sp_auto to data, predict, and inspect the tuned sp parameter
    >>> sp_auto.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    PluginParamsForecaster(...)
    >>> y_pred = sp_auto.predict()  # doctest: +SKIP
    >>> sp_auto.forecaster_.get_params()["sp"]  # doctest: +SKIP
    12
    >>> # shorthand ways to specify sp_auto, via dunder, does the same
    >>> sp_auto = sp_est * fcst  # doctest: +SKIP
    >>> # or entire pipeline in one go
    >>> sp_auto = Differencer() * SeasonalityACF() * NaiveForecaster()  # doctest: +SKIP

    using dictionary to plug "foo" parameter into "sp"

    >>> from sktime.param_est.fixed import FixedParams
    >>> sp_plugin = PluginParamsForecaster(
    ...     FixedParams({"foo": 12}), NaiveForecaster(), params={"sp": "foo"}
    ... )  # doctest: +SKIP
    """

    _tags = {
        "authors": "fkiraly",
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

        super().__init__()

        self._set_delegated_tags(self.forecaster_)

        # parameter estimators that are univariate do not broadcast,
        # so broadcasting needs to be done by the composite (i.e., self)
        if param_est.get_tags()["object_type"] == "param_est":
            if not param_est.get_tags()["capability:multivariate"]:
                self.set_tags(**{"scitype:y": "univariate"})

        self.set_tags(**{"fit_is_empty": False})
        # todo: only works for single series now
        #   think about how to deal with vectorization later
        self.set_tags(**{"X_inner_mtype": ["pd.DataFrame", "pd.Series", "np.ndarray"]})
        self.set_tags(**{"y_inner_mtype": ["pd.DataFrame", "pd.Series", "np.ndarray"]})

    def _fit(self, y, X, fh):
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

        # map args y, X, fh onto inner signature
        # y is passed always
        # X is passed if param_est fit has at least two arguments
        # fh is passed if any remaining argument is fh
        inner_params = list(signature(param_est.fit).parameters.keys())
        fit_kwargs = {}
        if len(inner_params) > 1:
            fit_kwargs[inner_params[1]] = X
            if "fh" in inner_params[2:]:
                fit_kwargs["fh"] = fh

        param_est.fit(y, **fit_kwargs)
        fitted_params = param_est.get_fitted_params()

        param_map = _resolve_param_map(param_est, forecaster, self.params)
        self.param_map_ = param_map

        # obtain the values of fitted params, and set forecaster to those
        new_params = {k: fitted_params[v] for k, v in param_map.items()}
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
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.param_est.fixed import FixedParams
        from sktime.param_est.seasonality import SeasonalityACF
        from sktime.utils.dependencies import _check_estimator_deps

        # use of dictionary to plug "foo" parameter into "sp", uses mock param_est
        params1 = {
            "forecaster": NaiveForecaster(),
            "param_est": FixedParams({"foo": 12}),
            "params": {"sp": "foo"},
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
