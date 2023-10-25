"""Test functions for data input and output."""

__author__ = ["SebasKoel", "Emiliathewolf", "TonyBagnall", "jasonlines", "achieveordie"]

__all__ = []


import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_basic_motions, load_UCR_UEA_dataset, load_uschange
from sktime.datasets._data_io import _list_available_datasets, _load_provided_dataset
from sktime.datatypes import check_is_mtype, scitype_to_mtype

# using this and not a direct import
# in order to avoid mtypes that require soft dependencies
MTYPE_LIST_PANEL = scitype_to_mtype("Panel")

# Disabling test for these mtypes since they don't support certain functionality yet
_TO_DISABLE = ["pd-long", "pd-wide", "numpyflat"]


@pytest.mark.parametrize("return_X_y", [True, False])
@pytest.mark.parametrize(
    "return_type", [mtype for mtype in MTYPE_LIST_PANEL if mtype not in _TO_DISABLE]
)
def test_load_provided_dataset(return_X_y, return_type):
    """Test function to check for proper loading.

    Check all possibilities of return_X_y and return_type.
    """
    if return_X_y:
        X, y = _load_provided_dataset("UnitTest", "TRAIN", return_X_y, return_type)
    else:
        X = _load_provided_dataset("UnitTest", "TRAIN", return_X_y, return_type)

    # Check whether object is same mtype or not, via bool
    valid, check_msg, _ = check_is_mtype(X, return_type, return_metadata=True)
    msg = (
        "load_basic_motions return has unexpected type on "
        f"return_X_y = {return_X_y}, return_type = {return_type}. "
        f"Error message returned by check_is_mtype: {check_msg}"
    )
    assert valid, msg


@pytest.mark.parametrize("return_X_y", [True, False])
@pytest.mark.parametrize(
    "return_type", [mtype for mtype in MTYPE_LIST_PANEL if mtype not in _TO_DISABLE]
)
def test_load_basic_motions(return_X_y, return_type):
    """Test load_basic_motions function to check for proper loading.

    Check all possibilities of return_X_y and return_type.
    """
    if return_X_y:
        X, y = load_basic_motions("TRAIN", return_X_y, return_type)
    else:
        X = load_basic_motions("TRAIN", return_X_y, return_type)

    # Check whether object is same mtype or not, via bool
    valid, check_msg, _ = check_is_mtype(X, return_type, return_metadata=True)
    msg = (
        "load_basic_motions return has unexpected type on "
        f"return_X_y = {return_X_y}, return_type = {return_type}. "
        f"Error message returned by check_is_mtype: {check_msg}"
    )
    assert valid, msg


def test_load_UCR_UEA_dataset():
    """Tests load_UCR_UEA_dataset correctly loads a baked in data set.

    Note this does not test whether download from timeseriesclassification.com works
    correctly, since this would make testing dependent on an external website.
    """
    X, y = load_UCR_UEA_dataset(name="UnitTest")
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray)
    assert X.shape == (42, 1) and y.shape == (42,)


_CHECKS = {
    "uschange": {
        "columns": ["Income", "Production", "Savings", "Unemployment"],
        "len_y": 187,
        "len_X": 187,
        "data_types_X": {
            "Income": "float64",
            "Production": "float64",
            "Savings": "float64",
            "Unemployment": "float64",
        },
        "data_type_y": "float64",
        "data": load_uschange(),
    },
}


@pytest.mark.parametrize("dataset", sorted(_CHECKS.keys()))
def test_data_loaders(dataset):
    """Assert if datasets are loaded correctly.

    dataset: dictionary with values to assert against should contain:
        'columns' : list with column names in correct order,
        'len_y'   : length of the y series (int),
        'len_X'   : length of the X series/dataframe (int),
        'data_types_X' : dictionary with column name keys and dtype as value,
        'data_type_y'  : dtype if y column (string)
        'data'    : tuple with y series and X series/dataframe if one is not
                    applicable fill with None value,
    """
    checks = _CHECKS[dataset]
    y = checks["data"][0]
    X = checks["data"][1]

    if y is not None:
        assert isinstance(y, pd.Series)
        assert len(y) == checks["len_y"]
        assert y.dtype == checks["data_type_y"]

    if X is not None:
        if len(checks["data_types_X"]) > 1:
            assert isinstance(X, pd.DataFrame)
        else:
            assert isinstance(X, pd.Series)

        assert X.columns.values.tolist() == checks["columns"]

        for col, dt in checks["data_types_X"].items():
            assert X[col].dtype == dt

        assert len(X) == checks["len_X"]


@pytest.mark.parametrize("origin_repo", [None, "forecastingorg"])
def test_list_available_datasets(origin_repo):
    """Test function for listing available datasets.

    check for two datasets repo format types:
    1. https://www.timeseriesclassification.com/
    2  https://forecastingdata.org/

    """
    dataset_name = "UnitTest"
    available_datasets = _list_available_datasets(
        extract_path=None, origin_repo=origin_repo
    )
    assert (
        dataset_name in available_datasets
    ), f"{dataset_name} dataset should be available."  # noqa: E501
