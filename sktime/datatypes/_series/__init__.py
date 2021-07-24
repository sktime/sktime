# -*- coding: utf-8 -*-
"""Module exports: Series type converters and mtype inference."""

from sktime.datatypes._series._convert import convert_dict as convert_dict_Series
from sktime.datatypes._series._mtypes import infer_mtype_dict as infer_mtype_dict_Series

from sktime.datatypes._series._examples import example_dict as example_dict_Series
from sktime.datatypes._series._examples import (
    example_dict_lossy as example_dict_lossy_Series,
)

from sktime.datatypes._series._registry import MTYPE_LIST_SERIES, MTYPE_REGISTER_SERIES

__all__ = [
    "convert_dict_Series",
    "infer_mtype_dict_Series",
    "MTYPE_LIST_SERIES",
    "MTYPE_REGISTER_SERIES",
    "example_dict_Series",
    "example_dict_lossy_Series",
]
