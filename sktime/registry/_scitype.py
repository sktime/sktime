# -*- coding: utf-8 -*-
"""Utility to determine scitype of estimator, based on base class type."""

__author__ = ["fkiraly"]

from inspect import isclass

from sktime.registry._base_classes import BASE_CLASS_REGISTER


def scitype(obj, force_single_scitype=True, coerce_to_list=False):
    """Determine scitype string of obj.

    Parameters
    ----------
    obj : class or object inheriting from sktime BaseObject
    force_single_scitype : bool, optional, default = True
        whether only a single scitype is returned
        if True, only the *first* scitype found will be returned
        order is determined by the order in BASE_CLASS_REGISTER
    coerce_to_list : bool, optional, default = False
        whether return should be coerced to list, even if only one scitype is identified

    Returns
    -------
    scitype : str, or list of str of sktime scitype strings from BASE_CLASS_REGISTER
        str, sktime scitype string, if exactly one scitype can be determined for obj
            or force_single_scitype is True, and if coerce_to_list is False
        list of str, of scitype strings, if more than one scitype are determined,
            or if coerce_to_list is True
        obj has scitype if it inherits from class in same row of BASE_CLASS_REGISTER

    Raises
    ------
    TypeError if no scitype can be determined for obj
    """
    if isclass(obj):
        scitypes = [sci[0] for sci in BASE_CLASS_REGISTER if issubclass(obj, sci[1])]
    else:
        scitypes = [sci[0] for sci in BASE_CLASS_REGISTER if isinstance(obj, sci[1])]

    if len(scitypes) == 0:
        raise TypeError("Error, no scitype could be determined for obj")

    if len(scitypes) > 1 and "object" in scitypes:
        scitypes = list(set(scitypes).difference(["object"]))

    if len(scitypes) > 1 and "estimator" in scitypes:
        scitypes = list(set(scitypes).difference(["estimator"]))

    if force_single_scitype:
        scitypes = [scitypes[0]]

    if len(scitypes) == 1 and not coerce_to_list:
        return scitypes[0]

    return scitypes
