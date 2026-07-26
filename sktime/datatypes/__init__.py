"""Module exports: data type definitions, checks, validation, fixtures, converters."""

from sktime.datatypes._check import (
    check_is_error_msg,
    check_is_mtype,
    check_is_scitype,
    check_raise,
    mtype,
    scitype,
)
from sktime.datatypes._convert import convert, convert_to
from sktime.datatypes._examples import get_examples
from sktime.datatypes._registry import (
    AMBIGUOUS_MTYPES,
    SCITYPE_LIST,
    SCITYPE_REGISTER,
    generate_mtype_list,
    generate_mtype_register,
    mtype_to_scitype,
    scitype_to_mtype,
)
from sktime.datatypes._utilities import get_cutoff, update_data
from sktime.datatypes._vectorize import VectorizedDF

__all__ = [
    "ALL_TIME_SERIES_MTYPES",
    "check_is_mtype",
    "check_is_scitype",
    "check_is_error_msg",
    "check_raise",
    "convert",
    "convert_to",
    "mtype",
    "get_cutoff",
    "get_examples",
    "mtype_to_scitype",
    "MTYPE_REGISTER",
    "MTYPE_LIST_HIERARCHICAL",
    "MTYPE_LIST_PANEL",
    "MTYPE_LIST_PROBA",
    "MTYPE_LIST_SERIES",
    "MTYPE_LIST_TABLE",
    "scitype",
    "scitype_to_mtype",
    "SCITYPE_LIST",
    "SCITYPE_REGISTER",
    "update_data",
    "VectorizedDF",
]


def __getattr__(name):
    getter_dict = {
        "MTYPE_LIST_ALIGNMENT": "Alignment",
        "MTYPE_LIST_HIERARCHICAL": "Hierarchical",
        "MTYPE_LIST_PANEL": "Panel",
        "MTYPE_LIST_PROBA": "Proba",
        "MTYPE_LIST_SERIES": "Series",
        "MTYPE_LIST_TABLE": "Table",
    }
    if name in getter_dict:
        return generate_mtype_list(scitype=getter_dict[name])

    if name == "MTYPE_REGISTER":
        return generate_mtype_register()

    if name == "ALL_TIME_SERIES_MTYPES":
        series = generate_mtype_list(scitype="Series")
        panel = generate_mtype_list(scitype="Panel")
        hierarchical = generate_mtype_list(scitype="Hierarchical")
        all_tsm = series + panel + hierarchical
        ALL_TIME_SERIES_MTYPES = list(set(all_tsm).difference(AMBIGUOUS_MTYPES))
        return ALL_TIME_SERIES_MTYPES

    raise AttributeError(f"module {__name__} has no attribute {name}")
