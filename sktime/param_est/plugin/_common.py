# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common routines for plugin estimators."""

__author__ = ["fkiraly"]


def _resolve_param_map(param_est, estimator, params=None):
    """Resolve parameter map from params for parameter plugin compositors.

    Parameters
    ----------
    param_est : sktime estimator object with a fit method, inheriting from BaseEstimator
        e.g., estimator inheriting from BaseParamFitter
        assumed to be fitted and have ``get_fitted_params`` method
    estimator : sktime object, inheriting from BaseObject or Basestimator
        assumed to have ``get_params`` method, not assumed to be fitted
    params : None, str, list of str, dict with str values/keys, optional, default=None
        determines which parameters from param_est are plugged into estimator and where
        None: all parameters of param_est are plugged into estimator
        only parameters present in both ``estimator`` and ``param_est`` are plugged in
        list of str: parameters in the list are plugged into parameters of the same name
        only parameters present in both ``estimator`` and ``param_est`` are plugged in
        str: considered as a one-element list of str with the string as single element
        dict: parameter with name of value is plugged into parameter with name of key
        only keys present in ``param_est`` and values in ``estimator`` are plugged in

    Returns
    -------
    param_map : dict with str keys and str values
        mapping of parameters from ``param_est_`` to estimator used in ``fit``,
        after filtering for parameters present in both
        to be used as ``param_map`` attribute of plugin compositor
    """
    fitted_params = param_est.get_fitted_params()

    # normalize params to a dict with str keys and str values
    if params is None:
        param_map = {x: x for x in fitted_params}
    elif isinstance(params, str):
        param_map = {params: params}
    elif isinstance(params, list):
        param_map = {x: x for x in params}
    elif isinstance(params, dict):
        param_map = params
    else:
        raise TypeError("params must be None, a str, a list of str, or a dict")

    # obtain the mapping restricted to param names that are available in both
    param_map = {x: param_map[x] for x in param_map if x in estimator.get_params()}
    param_map = {x: param_map[x] for x in param_map if param_map[x] in fitted_params}

    return param_map
