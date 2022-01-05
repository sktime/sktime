# -*- coding: utf-8 -*-
"""Utility to determine scitype of estimator, based on base class type."""

__author__ = ["fkiraly"]

from inspect import isclass

from sktime.registry._base_classes import BASE_CLASS_REGISTER


def scitype(obj):
    """Determine scitype string of obj.

    Parameters
    ----------
    obj : class or object inheriting from sktime BaseObject

    Returns
    -------
    scitype : str, or list of str of sktime scitype strings from BASE_CLASS_REGISTER
        str, sktime scitype string, if exactly one scitype can be determined for obj
        list of str, of scitype strings, if more than one scityp are determined
        obj has scitype if it inherits from class in same row of BASE_CLASS_REGISTER

    Raises
    ------
    TypeError if no scitype can be determined for obj
    """
    if isclass(obj):
        scitypes = [sci[0] for sci in BASE_CLASS_REGISTER if issubclass(obj, sci[1])]
    else:
        scitypes = [sci[0] for sci in BASE_CLASS_REGISTER if isinstance(obj, sci[1])]

    if len(scitypes) == 1:
        return scitypes[0]

    if len(scitypes) == 0:
        raise TypeError("Error, no scitype could be determined for obj")
