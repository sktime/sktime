"""Module exports: Series type checkers, converters and mtype inference."""

from sktime.datatypes._series._check import check_dict as check_dict_Series
from sktime.datatypes._series._convert import convert_dict as convert_dict_Series
from sktime.datatypes._series._examples import example_dict as example_dict_Series
from sktime.datatypes._series._examples import (
    example_dict_lossy as example_dict_lossy_Series,
)
from sktime.datatypes._series._examples import (
    example_dict_metadata as example_dict_metadata_Series,
)
from sktime.datatypes._series._registry import MTYPE_LIST_SERIES, MTYPE_REGISTER_SERIES

__all__ = [
    "check_dict_Series",
    "convert_dict_Series",
    "MTYPE_LIST_SERIES",
    "MTYPE_REGISTER_SERIES",
    "example_dict_Series",
    "example_dict_lossy_Series",
    "example_dict_metadata_Series",
]
