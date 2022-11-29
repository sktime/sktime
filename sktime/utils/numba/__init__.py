# -*- coding: utf-8 -*-
"""Numba utility functionality."""

from sktime.utils.validation._dependencies import _check_soft_dependencies

if not _check_soft_dependencies("numba", severity="none"):
    __all__ = []
