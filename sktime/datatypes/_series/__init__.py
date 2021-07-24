# -*- coding: utf-8 -*-
"""Module exports: Series type converters and mtype inference."""

__all__ = ["convert_dict_Series", "infer_mtype_dict_Series"]

from sktime.datatypes._series._convert import convert_dict as convert_dict_Series
from sktime.datatypes._series._mtypes import infer_mtype_dict as infer_mtype_dict_Series
