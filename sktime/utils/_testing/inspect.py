#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "_get_args"
]

from inspect import signature


def _get_args(function, varargs=False):
    """Helper to get function arguments"""
    try:
        params = signature(function).parameters
    except ValueError:
        # Error on builtin C function
        return []
    args = [key for key, param in params.items()
            if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
    if varargs:
        varargs = [param.name for param in params.values()
                   if param.kind == param.VAR_POSITIONAL]
        if len(varargs) == 0:
            varargs = None
        return args, varargs
    else:
        return args
