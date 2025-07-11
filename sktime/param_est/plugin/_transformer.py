# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Plugin composite for substituting parameter estimator fit into transformers."""

__author__ = ["fkiraly"]
__all__ = ["PluginParamsTransformer"]

from inspect import signature

from sktime.param_est.plugin._common import _resolve_param_map
from sktime.transformations._delegate import _DelegatedTransformer


class PluginParamsTransformer(_DelegatedTransformer):
    """Plugs parameters from a parameter estimator into a transformer.

    In ``fit``, first fits ``param_est`` to data passed:

    * ``X`` of ``fit`` is passed as the first arg to ``param_est.fit``
    * ``y`` of ``fit`` is passed as the second arg to ``param_est.fit``,
      if ``param_est.fit`` has a second arg

    Then, does ``transformer.set_params`` with desired/selected parameters.
    Parameters of the fitted ``param_est`` are passed on to ``transformer``,
    from/to pairs are as specified by the ``params`` parameter of ``self``, see below.

    Then, fits ``transformer`` to the data passed in ``fit``.

    After that, behaves identically to ``transformer`` with those parameters set.

    Example: ``param_est`` seasonality test to determine ``sp`` parameter;
    ``transformer`` a transformer with an ``sp`` parameter, e.g., ``Deseasonalizer``.

    Parameters
    ----------
    param_est : sktime estimator object with a fit method, inheriting from BaseEstimator
        e.g., estimator inheriting from BaseParamFitter or transformer
        this is a "blueprint" estimator, state does not change when ``fit`` is called

    transformer : sktime transformer, i.e., estimator inheriting from BaseTransformer
        this is a "blueprint" estimator, state does not change when ``fit`` is called

    params : None, str, list of str, dict with str values/keys, optional, default=None
        determines which parameters from ``param_est`` are plugged into trafo and where
        None: all parameters of param_est are plugged into transformer
        only parameters present in both ``transformer`` and ``param_est`` are plugged in
        list of str: parameters in the list are plugged into parameters of the same name
        only parameters present in both ``transformer`` and ``param_est`` are plugged in
        str: considered as a one-element list of str with the string as single element
        dict: parameter with name of value is plugged into parameter with name of key
        only keys present in ``param_est`` and values in ``transformer`` are plugged in

    Attributes
    ----------
    param_est_ : sktime parameter estimator, clone of estimator in ``param_est``
        this clone is fitted in the pipeline when ``fit`` is called
    transformer_ : sktime transformer, clone of ``transformer``
        this clone is fitted in the pipeline when ``fit`` is called
    param_map_ : dict
        mapping of parameters from ``param_est_`` to ``transformer_`` used in ``fit``,
        after filtering for parameters present in both

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.param_est.plugin import PluginParamsTransformer
    >>> from sktime.param_est.seasonality import SeasonalityACF
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.transformations.series.difference import Differencer
    >>>
    >>> X = load_airline()  # doctest: +SKIP
    >>>
    >>> # sp_est is a seasonality estimator
    >>> # ACF assumes stationarity so we concat with differencing first
    >>> sp_est = Differencer() * SeasonalityACF()  # doctest: +SKIP

    >>> # trafo is a forecaster with a "sp" parameter which we want to tune
    >>> trafo = Deseasonalizer()  # doctest: +SKIP
    >>> sp_auto = PluginParamsTransformer(sp_est, trafo)  # doctest: +SKIP
    >>>
    >>> # fit sp_auto to data, transform, and inspect the tuned sp parameter
    >>> sp_auto.fit(X)  # doctest: +SKIP
    PluginParamsTransformer(...)
    >>> Xt = sp_auto.transform(X)  # doctest: +SKIP
    >>> sp_auto.transformer_.get_params()["sp"]  # doctest: +SKIP
    12
    >>> # shorthand ways to specify sp_auto, via dunder, does the same
    >>> sp_auto = sp_est * trafo  # doctest: +SKIP
    >>> # or entire pipeline in one go
    >>> sp_auto = Differencer() * SeasonalityACF() * Deseasonalizer()  # doctest: +SKIP

    using dictionary to plug "foo" parameter into "sp"

    >>> from sktime.param_est.fixed import FixedParams
    >>> sp_plugin = PluginParamsTransformer(
    ...     FixedParams({"foo": 12}), Deseasonalizer(), params={"sp": "foo"}
    ... )  # doctest: +SKIP
    """

    _tags = {
        "authors": "fkiraly",
        "capability:inverse_transform": True,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "remember_data": False,  # whether all data seen is remembered as self._X
    }

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods to those of same name in self.transformer_
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    def __init__(self, param_est, transformer, params=None, update_params=False):
        self.param_est = param_est
        self.param_est_ = param_est.clone()
        self.transformer = transformer
        self.transformer_ = transformer.clone()
        self.params = params
        self.update_params = update_params

        super().__init__()

        TAGS_TO_CLONE = [
            "scitype:transform-input",
            "scitype:transform-output",
            "scitype:transform-labels",
            "scitype:instancewise",
            "capability:inverse_transform",
            "capability:inverse_transform:range",
            "capability:inverse_transform:exact",
            "univariate-only",
            "y_inner_mtype",
            "requires_y",
            "enforce_index_type",
            "X-y-must-have-same-index",
            "transform-returns-same-time-index",
            "skip-inverse-transform",
            "capability:unequal_length",
            "capability:unequal_length:removes",
            "capability:missing_values",
            "capability:missing_values:removes",
        ]

        self.clone_tags(self.transformer_, TAGS_TO_CLONE)

        # todo: only works for single series now
        #   think about how to deal with vectorization later
        SERIES_MTYPES = ["pd.DataFrame", "pd.Series", "np.ndarray"]
        self.set_tags(**{"X_inner_mtype": SERIES_MTYPES})

        if self.get_tag("y_inner_mtype") not in [None, "None"]:
            self.set_tags(**{"y_inner_mtype": SERIES_MTYPES})

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        # reference to delegate
        transformer = self._get_delegate()

        # fit the parameter estimator to X
        param_est = self.param_est_

        # map args X, y onto inner signature
        # X is passed always
        # y is passed if param_est fit has at least two arguments
        inner_params = list(signature(param_est.fit).parameters.keys())
        fit_kwargs = {}
        if len(inner_params) > 1:
            fit_kwargs[inner_params[1]] = y

        param_est.fit(X, **fit_kwargs)
        fitted_params = param_est.get_fitted_params()

        param_map = _resolve_param_map(param_est, transformer, self.params)
        self.param_map_ = param_map

        # obtain the values of fitted params, and set transformer to those
        new_params = {k: fitted_params[v] for k, v in param_map.items()}
        transformer.set_params(**new_params)

        # fit the transformer, with the fitted parameter values
        transformer.fit(X=X, y=y)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

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
        from sktime.param_est.fixed import FixedParams
        from sktime.param_est.seasonality import SeasonalityACF
        from sktime.transformations.series.detrend import Deseasonalizer
        from sktime.transformations.series.exponent import ExponentTransformer
        from sktime.utils.dependencies import _check_estimator_deps

        # use of dictionary to plug "foo" parameter into "power", uses mock param_est
        params1 = {
            "transformer": ExponentTransformer(),
            "param_est": FixedParams({"foo": 12}),
            "params": {"power": "foo"},
        }
        params = [params1]

        # uses a "real" param est that depends on statsmodels, requires statsmodels
        if _check_estimator_deps(SeasonalityACF, severity="none"):
            # explicit reference to a parameter "sp", present in both estimators
            params2 = {
                "transformer": Deseasonalizer(),
                "param_est": SeasonalityACF(),
                "params": "sp",
            }
            params = params + [params2]

        return params
