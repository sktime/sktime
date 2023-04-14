# -*- coding: utf-8 -*-
"""Utilities to dispatch on estimators."""

__author__ = ["fkiraly"]


def construct_dispatch(cls, params=None):
    """Construct an estimator with an overspecified parameter dictionary.

    Constructs and returns an instance of `cls`, using parameters in a dict `params`.
    The dict `params` may contain keys that `cls` does not have, which are ignored.

    This is useful in multiplexing or dispatching over multiple `cls` which have
    different and potentially intersecting parameter sets.

    Parameters
    ----------
    cls : sktime estimator, inheriting from `BaseObject`
    params : dict with str keys, optional, default = None = {}

    Examples
    --------
    >>> from sktime.utils.estimators import construct_dispatch
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> params = {"strategy": "drift", "foo": "bar", "bar": "foo"}
    >>> construct_dispatch(NaiveForecaster, params)
    NaiveForecaster(strategy='drift')
    """
    cls_param_names = cls.get_param_names()
    cls_params_in_dict = set(cls_param_names).intersection(params.keys())
    params_for_cls = {key: params[key] for key in cls_params_in_dict}

    obj = cls(**params_for_cls)
    return obj
