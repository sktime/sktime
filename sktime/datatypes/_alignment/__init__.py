# -*- coding: utf-8 -*-
"""Module exports: Alignment type checkers and mtype inference."""

from sktime.datatypes._alignment._check import check_dict as check_dict_Alignment
from sktime.datatypes._alignment._examples import example_dict as example_dict_Alignment
from sktime.datatypes._alignment._registry import (
    MTYPE_LIST_ALIGNMENT,
    MTYPE_REGISTER_ALIGNMENT,
)

__all__ = [
    "check_dict_Alignment",
    "example_dict_Alignment",
    "MTYPE_LIST_ALIGNMENT",
    "MTYPE_REGISTER_ALIGNMENT",
]
