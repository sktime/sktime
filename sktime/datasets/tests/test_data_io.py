# -*- coding: utf-8 -*-
"""Test functions for data input and output."""

__author__ = ["SebasKoel", "Emiliathewolf", "TonyBagnall", "jasonlines", "achieveordie"]

__all__ = []

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

import sktime
from sktime.datasets import (
    generate_example_long_table,
    load_basic_motions,
    load_from_long_to_dataframe,
    load_from_tsfile,
    load_from_tsfile_to_dataframe,
    load_solar,
    load_tsf_to_dataframe,
    load_UCR_UEA_dataset,
    load_uschange,
    write_dataframe_to_tsfile,
)
from sktime.datasets._data_io import (
    MODULE,
    _convert_tsf_to_hierarchical,
    _load_provided_dataset,
)
from sktime.datatypes import MTYPE_LIST_PANEL, check_is_mtype
from sktime.utils.validation._dependencies import _check_soft_dependencies

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


def test_load_from_tsfile():
    """Test function for loading TS formats.

    Test
    1. Univariate equal length (UnitTest) returns 2D numpy X, 1D numpy y
    2. Multivariate equal length (BasicMotions) returns 3D numpy X, 1D numpy y
    3. Univariate and multivariate unequal length (PLAID) return X as DataFrame
    """
    data_path = MODULE + "/data/UnitTest/UnitTest_TRAIN.ts"
    # Test 1.1: load univariate equal length (UnitTest), should return 2D array and 1D
    # array, test first and last data
    # Test 1.2: Load a problem without y values (UnitTest),  test first and last data.
    X, y = load_from_tsfile(data_path, return_data_type="np2D")
    X2 = load_from_tsfile(data_path, return_y=False, return_data_type="np2D")
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.ndim == 2 and X2.ndim == 2
    assert X.shape == (20, 24) and y.shape == (20,)
    assert X[0][0] == 573.0
    X2 = load_from_tsfile(data_path, return_y=False, return_data_type="numpy3D")
    assert isinstance(X2, np.ndarray)
    assert X2.ndim == 3
    assert X2.shape == (20, 1, 24)
    assert X2[0][0][0] == 573.0

    # Test 2: load multivare equal length (BasicMotions), should return 3D array and 1D
    # array, test first and last data.
    data_path = MODULE + "/data/BasicMotions/BasicMotions_TRAIN.ts"
    X, y = load_from_tsfile(data_path, return_data_type="numpy3d")
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape == (40, 6, 100) and y.shape == (40,)
    assert X[1][2][3] == -1.898794
    X, y = load_from_tsfile(data_path)
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray)
    assert X.shape == (40, 6) and y.shape == (40,)
    assert isinstance(X.iloc[1, 2], pd.Series)
    assert X.iloc[1, 2].iloc[3] == -1.898794

    # Test 3.1: load univariate unequal length (PLAID), should return a one column
    # dataframe,
    data_path = MODULE + "/data/PLAID/PLAID_TRAIN.ts"
    X, y = load_from_tsfile(full_file_path_and_name=data_path)
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray)
    assert X.shape == (537, 1) and y.shape == (537,)
    # Test 3.2: load multivariate unequal length (JapaneseVowels), should return a X
    # columns dataframe,
    data_path = MODULE + "/data/JapaneseVowels/JapaneseVowels_TRAIN.ts"
    X, y = load_from_tsfile(full_file_path_and_name=data_path)
    assert isinstance(X, pd.DataFrame) and isinstance(y, np.ndarray)
    assert X.shape == (270, 12) and y.shape == (270,)


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
    """
    Assert if datasets are loaded correctly.

    dataset: dictionary with values to assert against should contain:
        'columns' : list with column names in correct order,
        'len_y'   : lenght of the y series (int),
        'len_X'   : lenght of the X series/dataframe (int),
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


def test_load_from_tsfile_to_dataframe():
    """Test the load_from_tsfile_to_dataframe() function."""
    # Test that an empty file is classed an invalid
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            # Write the contents of the file
            file_contents = ""
            tmp_file.write(file_contents)
            tmp_file.flush()
            # Parse the file and assert that it is invalid
            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)
    finally:
        os.remove(path)
    # Test that a file with an incomplete set of metadata is invalid
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            # Write the contents of the file
            file_contents = (
                "@problemName Test Problem\n@timeStamps " "true\n@univariate true\n"
            )
            tmp_file.write(file_contents)
            tmp_file.flush()
            # Parse the file and assert that it is invalid
            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)
    finally:
        os.remove(path)
    # Test that a file with a complete set of metadata but no data is invalid
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            # Write the contents of the file
            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel false\n@data"
            )
            tmp_file.write(file_contents)
            tmp_file.flush()
            # Parse the file and assert that it is invalid
            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)
    finally:
        os.remove(path)
    # Test that a file with a complete set of metadata and no data but
    # invalid metadata values is invalid
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            # Write the contents of the file
            file_contents = (
                "@problemName\n@timeStamps\n@univariate "
                "true\n@classLabel false\n@data"
            )
            tmp_file.write(file_contents)
            tmp_file.flush()
            # Parse the file and assert that it is invalid
            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)
    finally:
        os.remove(path)
    # Test that a file with a complete set of metadata and a single
    # case/dimension parses correctly
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            # Write the contents of the file
            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += "(0, 1), (1, 2)"

            tmp_file.write(file_contents)
            tmp_file.flush()
            # Parse the file
            df = load_from_tsfile_to_dataframe(path)
            # Test the DataFrame returned accurately reflects the data in
            # the file
            np.testing.assert_equal(len(df), 1)
            np.testing.assert_equal(len(df.columns), 1)
            series = df["dim_0"]
            np.testing.assert_equal(len(series), 1)
            series = df["dim_0"][0]
            np.testing.assert_equal(series[0], 1.0)
            np.testing.assert_equal(series[1], 2.0)
    finally:
        os.remove(path)
    # Test that a file with a complete set of metadata and 2 cases with 3
    # dimensions parses correctly
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            # Write the contents of the file
            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += "(0, 1), (1, 2):(0, 3), (1, 4):(0, 5), (1, 6)\n"
            file_contents += "(0, 11), (1, 12):(0, 13), (1,14):(0, 15), (1, 16)     \n"
            tmp_file.write(file_contents)
            tmp_file.flush()
            # Parse the file
            df = load_from_tsfile_to_dataframe(path)
            # Test the DataFrame returned accurately reflects the data in
            # the file
            np.testing.assert_equal(len(df), 2)
            np.testing.assert_equal(len(df.columns), 3)
            series = df["dim_0"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_0"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 1.0)
            np.testing.assert_equal(series[1], 2.0)

            series = df["dim_0"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 11.0)
            np.testing.assert_equal(series[1], 12.0)

            series = df["dim_1"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_1"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 3.0)
            np.testing.assert_equal(series[1], 4.0)

            series = df["dim_1"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 13.0)
            np.testing.assert_equal(series[1], 14.0)

            series = df["dim_2"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_2"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 5.0)
            np.testing.assert_equal(series[1], 6.0)

            series = df["dim_2"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 15.0)
            np.testing.assert_equal(series[1], 16.0)
    finally:
        os.remove(path)
    # Test that a file with a complete set of metadata and time-series of
    # different length parses correctly
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            # Write the contents of the file
            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += "(0, 1), (1, 2):(0, 3):(0, 5), (1, 6)\n"
            file_contents += "(0, 11), (1, 12):(0, 13), (1,14):(0, 15)\n"
            tmp_file.write(file_contents)
            tmp_file.flush()
            # Parse the file
            df = load_from_tsfile_to_dataframe(path)
            # Test the DataFrame returned accurately reflects the data in
            # the file

            np.testing.assert_equal(len(df), 2)
            np.testing.assert_equal(len(df.columns), 3)

            series = df["dim_0"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_0"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 1.0)
            np.testing.assert_equal(series[1], 2.0)

            series = df["dim_0"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 11.0)
            np.testing.assert_equal(series[1], 12.0)

            series = df["dim_1"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_1"][0]
            np.testing.assert_equal(len(series), 1)
            np.testing.assert_equal(series[0], 3.0)

            series = df["dim_1"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 13.0)
            np.testing.assert_equal(series[1], 14.0)

            series = df["dim_2"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_2"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 5.0)
            np.testing.assert_equal(series[1], 6.0)

            series = df["dim_2"][1]
            np.testing.assert_equal(len(series), 1)
            np.testing.assert_equal(series[0], 15.0)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data but an
    # inconsistent number of dimensions across cases is classed as invalid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += "(0, 1), (1, 2):(0, 3), (1, 4):(0, 5), (1, 6)\n"
            file_contents += "(0, 11), (1, 12):(0, 13), (1,14)    \n"

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file and assert that it is invalid

            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data but missing
    # values after a tuple is classed as invalid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += "(0, 1), (1, 2):(0, 3), (1, 4):(0, 5),\n"

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file and assert that it is invalid

            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data and some
    # empty dimensions is classed as valid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += "(0, 1), (1, 2):     :(0, 5), (1, 6)\n"
            file_contents += "(0, 11), (1, 12):(0, 13), (1,14)    :       \n"
            file_contents += (
                "(0, 21), (1, 22):(0, 23), (1,24)    :   (0,25), (1, 26)    \n"
            )

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file

            df = load_from_tsfile_to_dataframe(path)

            # Test the DataFrame returned accurately reflects the data in
            # the file

            np.testing.assert_equal(len(df), 3)
            np.testing.assert_equal(len(df.columns), 3)

            series = df["dim_0"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_0"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 1.0)
            np.testing.assert_equal(series[1], 2.0)

            series = df["dim_0"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 11.0)
            np.testing.assert_equal(series[1], 12.0)

            series = df["dim_0"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 21.0)
            np.testing.assert_equal(series[1], 22.0)

            series = df["dim_1"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_1"][0]
            np.testing.assert_equal(len(series), 0)

            series = df["dim_1"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 13.0)
            np.testing.assert_equal(series[1], 14.0)

            series = df["dim_1"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 23.0)
            np.testing.assert_equal(series[1], 24.0)

            series = df["dim_2"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_2"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 5.0)
            np.testing.assert_equal(series[1], 6.0)

            series = df["dim_2"][1]
            np.testing.assert_equal(len(series), 0)

            series = df["dim_2"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 25.0)
            np.testing.assert_equal(series[1], 26.0)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data that
    # contains datetimes as timestamps and has some empty dimensions is
    # classed as valid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += (
                "(01/01/2019 00:00:00, 1),  (01/02/2019 "
                "00:00:00, 2)  :                               "
                "                      : (01/05/2019 00:00:00, "
                "5), (01/06/2019 00:00:00, 6)\n"
            )
            file_contents += (
                "(01/01/2020 00:00:00, 11), (01/02/2020 "
                "00:00:00, 12) : (01/03/2020 00:00:00, 13), "
                "(01/04/2020 00:00:00, 14) :  \n"
            )
            file_contents += (
                "(01/01/2021 00:00:00, 21), (01/02/2021 "
                "00:00:00, 22) : (01/03/2021 00:00:00, 23), "
                "(01/04/2021 00:00:00, 24) :  \n"
            )

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file

            df = load_from_tsfile_to_dataframe(path)

            # Test the DataFrame returned accurately reflects the data in
            # the file

            np.testing.assert_equal(len(df), 3)
            np.testing.assert_equal(len(df.columns), 3)

            series = df["dim_0"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_0"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series["01/01/2019"], 1.0)
            np.testing.assert_equal(series["01/02/2019"], 2.0)

            series = df["dim_0"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series["01/01/2020"], 11.0)
            np.testing.assert_equal(series["01/02/2020"], 12.0)

            series = df["dim_0"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series["01/01/2021"], 21.0)
            np.testing.assert_equal(series["01/02/2021"], 22.0)

            series = df["dim_1"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_1"][0]
            np.testing.assert_equal(len(series), 0)

            series = df["dim_1"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series["01/03/2020"], 13.0)
            np.testing.assert_equal(series["01/04/2020"], 14.0)

            series = df["dim_1"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series["01/03/2021"], 23.0)
            np.testing.assert_equal(series["01/04/2021"], 24.0)

            series = df["dim_2"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_2"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series["01/05/2019"], 5.0)
            np.testing.assert_equal(series["01/06/2019"], 6.0)

            series = df["dim_2"][1]
            np.testing.assert_equal(len(series), 0)

            series = df["dim_2"][2]
            np.testing.assert_equal(len(series), 0)

    finally:
        os.remove(path)

    # Test that a file that mixes timestamp conventions is invalid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += (
                "(01/01/2019 00:00:00, 1),  (01/02/2019 "
                "00:00:00, 2)  :                               "
                "                      : (01/05/2019 00:00:00, "
                "5), (01/06/2019 00:00:00, 6)\n"
            )
            file_contents += (
                "(00, 11), (1, 12) : (01/03/2020 00:00:00, 13), "
                "(01/04/2020 00:00:00, 14) :  \n"
            )
            file_contents += (
                "(01/01/2021 00:00:00, 21), (01/02/2021 "
                "00:00:00, 22) : (01/03/2021 00:00:00, 23), "
                "(01/04/2021 00:00:00, 24) :  \n"
            )

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file and assert that it is invalid

            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data but missing
    # classes is classed as invalid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel true 0 1 "
                "2\n@data\n"
            )
            file_contents += "(0, 1), (1, 2):(0, 3), (1, 4):(0, 5), (1, 6)\n"
            file_contents += "(0, 11), (1, 12):(0, 13), (1,14):(0, 15), (1, 16)     \n"

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file and assert that it is invalid

            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data but invalid
    # classes is classed as invalid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel true 0 1 "
                "2\n@data\n"
            )
            file_contents += "(0, 1), (1, 2):(0, 3), (1, 4):(0, 5), (1, 6) : 0 \n"
            file_contents += (
                "(0, 11), (1, 12):(0, 13), (1,14):(0, 15), (1, 16)   : 3  \n"
            )

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file and assert that it is invalid

            np.testing.assert_raises(IOError, load_from_tsfile_to_dataframe, path)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data with classes
    # is classed as valid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "true\n@univariate true\n@classLabel true 0 1 "
                "2\n@data\n"
            )
            file_contents += "(0, 1), (1, 2):(0, 3), (1, 4):(0, 5), (1, 6): 0\n"
            file_contents += (
                "(0, 11), (1, 12):(0, 13), (1,14):(0, 15), (1, 16): 2     \n"
            )

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file

            df, y = load_from_tsfile_to_dataframe(path)

            # Test the DataFrame of X values returned accurately reflects
            # the data in the file

            np.testing.assert_equal(len(df), 2)
            np.testing.assert_equal(len(df.columns), 3)

            series = df["dim_0"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_0"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 1.0)
            np.testing.assert_equal(series[1], 2.0)

            series = df["dim_0"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 11.0)
            np.testing.assert_equal(series[1], 12.0)

            series = df["dim_1"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_1"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 3.0)
            np.testing.assert_equal(series[1], 4.0)

            series = df["dim_1"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 13.0)
            np.testing.assert_equal(series[1], 14.0)

            series = df["dim_2"]
            np.testing.assert_equal(len(series), 2)

            series = df["dim_2"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 5.0)
            np.testing.assert_equal(series[1], 6.0)

            series = df["dim_2"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 15.0)
            np.testing.assert_equal(series[1], 16.0)

            # Test that the class values are as expected

            np.testing.assert_equal(len(y), 2)
            np.testing.assert_equal(y[0], "0")
            np.testing.assert_equal(y[1], "2")

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data, with no
    # timestamps, is classed as valid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "false\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += "1,2:3,4:5,6\n"
            file_contents += "11,12:13,14:15,16\n"
            file_contents += "21,22:23,24:25,26\n"

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file

            df = load_from_tsfile_to_dataframe(path)

            # Test the DataFrame returned accurately reflects the data in
            # the file

            np.testing.assert_equal(len(df), 3)
            np.testing.assert_equal(len(df.columns), 3)

            series = df["dim_0"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_0"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 1.0)
            np.testing.assert_equal(series[1], 2.0)

            series = df["dim_0"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 11.0)
            np.testing.assert_equal(series[1], 12.0)

            series = df["dim_0"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 21.0)
            np.testing.assert_equal(series[1], 22.0)

            series = df["dim_1"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_1"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 3.0)
            np.testing.assert_equal(series[1], 4.0)

            series = df["dim_1"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 13.0)
            np.testing.assert_equal(series[1], 14.0)

            series = df["dim_1"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 23.0)
            np.testing.assert_equal(series[1], 24.0)

            series = df["dim_2"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_2"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 5.0)
            np.testing.assert_equal(series[1], 6.0)

            series = df["dim_2"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 15.0)
            np.testing.assert_equal(series[1], 16.0)

            series = df["dim_2"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 25.0)
            np.testing.assert_equal(series[1], 26.0)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data, with no
    # timestamps and some empty dimensions, is classed as valid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:

            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "false\n@univariate true\n@classLabel "
                "false\n@data\n"
            )
            file_contents += "1,2::5,6\n"
            file_contents += "11,12:13,14:15,16\n"
            file_contents += "21,22:23,24:\n"

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file

            df = load_from_tsfile_to_dataframe(path)

            # Test the DataFrame returned accurately reflects the data in
            # the file

            np.testing.assert_equal(len(df), 3)
            np.testing.assert_equal(len(df.columns), 3)

            series = df["dim_0"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_0"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 1.0)
            np.testing.assert_equal(series[1], 2.0)

            series = df["dim_0"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 11.0)
            np.testing.assert_equal(series[1], 12.0)

            series = df["dim_0"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 21.0)
            np.testing.assert_equal(series[1], 22.0)

            series = df["dim_1"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_1"][0]
            np.testing.assert_equal(len(series), 0)

            series = df["dim_1"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 13.0)
            np.testing.assert_equal(series[1], 14.0)

            series = df["dim_1"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 23.0)
            np.testing.assert_equal(series[1], 24.0)

            series = df["dim_2"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_2"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 5.0)
            np.testing.assert_equal(series[1], 6.0)

            series = df["dim_2"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 15.0)
            np.testing.assert_equal(series[1], 16.0)

            series = df["dim_2"][2]
            np.testing.assert_equal(len(series), 0)

    finally:
        os.remove(path)

    # Test that a file with a complete set of metadata and data, with no
    # timestamps and some empty dimensions and classes, is classed as valid

    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            # Write the contents of the file

            file_contents = (
                "@problemName Test Problem\n@timeStamps "
                "false\n@univariate true\n@classLabel true cat "
                "bear dog\n@data\n"
            )
            file_contents += "1,2::5,6:cat  \n"
            file_contents += "11,12:13,14:15,16:  dog\n"
            file_contents += "21,22:23,24::   bear   \n"

            tmp_file.write(file_contents)
            tmp_file.flush()

            # Parse the file

            df, y = load_from_tsfile_to_dataframe(path)

            # Test the DataFrame of X values returned accurately reflects
            # the data in the file

            np.testing.assert_equal(len(df), 3)
            np.testing.assert_equal(len(df.columns), 3)

            series = df["dim_0"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_0"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 1.0)
            np.testing.assert_equal(series[1], 2.0)

            series = df["dim_0"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 11.0)
            np.testing.assert_equal(series[1], 12.0)

            series = df["dim_0"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 21.0)
            np.testing.assert_equal(series[1], 22.0)

            series = df["dim_1"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_1"][0]
            np.testing.assert_equal(len(series), 0)

            series = df["dim_1"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 13.0)
            np.testing.assert_equal(series[1], 14.0)

            series = df["dim_1"][2]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 23.0)
            np.testing.assert_equal(series[1], 24.0)

            series = df["dim_2"]
            np.testing.assert_equal(len(series), 3)

            series = df["dim_2"][0]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 5.0)
            np.testing.assert_equal(series[1], 6.0)

            series = df["dim_2"][1]
            np.testing.assert_equal(len(series), 2)
            np.testing.assert_equal(series[0], 15.0)
            np.testing.assert_equal(series[1], 16.0)

            series = df["dim_2"][2]
            np.testing.assert_equal(len(series), 0)

            # Test that the class values are as expected

            np.testing.assert_equal(len(y), 3)
            np.testing.assert_equal(y[0], "cat")
            np.testing.assert_equal(y[1], "dog")
            np.testing.assert_equal(y[2], "bear")

    finally:
        os.remove(path)


def test_load_from_long_to_dataframe(tmpdir):
    """Test for loading from long to dataframe."""
    # create and save a example long-format file to csv
    test_dataframe = generate_example_long_table()
    dataframe_path = tmpdir.join("data.csv")
    test_dataframe.to_csv(dataframe_path, index=False)
    # load and convert the csv to sktime-formatted data
    nested_dataframe = load_from_long_to_dataframe(dataframe_path)
    assert isinstance(nested_dataframe, pd.DataFrame)


def test_load_from_long_incorrect_format(tmpdir):
    """Test for loading from long with incorrect format."""
    with pytest.raises(ValueError):
        dataframe = generate_example_long_table()
        dataframe.drop(dataframe.columns[[3]], axis=1, inplace=True)
        dataframe_path = tmpdir.join("data.csv")
        dataframe.to_csv(dataframe_path, index=False)
        load_from_long_to_dataframe(dataframe_path)


@pytest.mark.parametrize("dataset", ["ItalyPowerDemand", "BasicMotions"])
def test_write_dataframe_to_ts_success(tmp_path, dataset):
    """Tests whether a dataset can be written by the .ts writer then read in."""
    # load an example dataset
    path = os.path.join(
        os.path.dirname(sktime.__file__),
        f"datasets/data/{dataset}/{dataset}_TEST.ts",
    )
    test_X, test_y = load_from_tsfile_to_dataframe(path)
    # output the dataframe in a ts file
    write_dataframe_to_tsfile(
        data=test_X,
        path=tmp_path,
        problem_name=dataset,
        class_label=np.unique(test_y),
        class_value_list=test_y,
        comment="""
          The data was derived from twelve monthly electrical power demand
          time series from Italy and first used in the paper "Intelligent
          Icons: Integrating Lite-Weight Data Mining and Visualization into
          GUI Operating Systems". The classification task is to distinguish
          days from Oct to March (inclusive) from April to September.
        """,
        fold="_transform",
    )
    # load data back from the ts file
    result = f"{tmp_path}/{dataset}/{dataset}_transform.ts"
    res_X, res_y = load_from_tsfile_to_dataframe(result)
    # check if the dataframes are the same
    assert_frame_equal(res_X, test_X)


def test_write_dataframe_to_ts_fail(tmp_path):
    """Tests if non-dataframes are handled correctly."""
    with pytest.raises(ValueError, match="Data provided must be a DataFrame"):
        write_dataframe_to_tsfile(
            data=np.random.rand(3, 2),
            path=str(tmp_path),
            problem_name="GunPoint",
        )


@pytest.mark.parametrize(
    "input_path, return_type, output_df",
    [
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader.tsf",
            "default_tsf",
            pd.DataFrame(
                {
                    "series_name": ["T1", "T2", "T3"],
                    "start_timestamp": [
                        pd.Timestamp(year=1979, month=1, day=1),
                        pd.Timestamp(year=1979, month=1, day=1),
                        pd.Timestamp(year=1973, month=1, day=1),
                    ],
                    "series_value": [
                        [
                            25092.2284,
                            24271.5134,
                            25828.9883,
                            27697.5047,
                            27956.2276,
                            29924.4321,
                            30216.8321,
                        ],
                        [887896.51, 887068.98, 971549.04],
                        [227921, 230995, 183635, 238605, 254186],
                    ],
                }
            ),
        ),
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader_hierarchical.tsf",
            "pd_multiindex_hier",
            pd.DataFrame(
                data=[
                    25092.2284,
                    24271.5134,
                    25828.9883,
                    27697.5047,
                    27956.2276,
                    29924.4321,
                    30216.8321,
                    887896.51,
                    887068.98,
                    971549.04,
                    227921,
                    230995,
                    183635,
                    238605,
                    254186,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("G1", "T1", pd.Timestamp(year=1979, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1980, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1981, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1982, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1983, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1984, month=1, day=1)),
                        ("G1", "T1", pd.Timestamp(year=1985, month=1, day=1)),
                        ("G1", "T2", pd.Timestamp(year=1979, month=1, day=1)),
                        ("G1", "T2", pd.Timestamp(year=1980, month=1, day=1)),
                        ("G1", "T2", pd.Timestamp(year=1981, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1973, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1974, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1975, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1976, month=1, day=1)),
                        ("G2", "T3", pd.Timestamp(year=1977, month=1, day=1)),
                    ],
                    names=["series_group", "series_name", "timestamp"],
                ),
                columns=["series_value"],
            ),
        ),
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader.tsf",
            "pd-multiindex",
            pd.DataFrame(
                data=[
                    25092.2284,
                    24271.5134,
                    25828.9883,
                    27697.5047,
                    27956.2276,
                    29924.4321,
                    30216.8321,
                    887896.51,
                    887068.98,
                    971549.04,
                    227921,
                    230995,
                    183635,
                    238605,
                    254186,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("T1", pd.Timestamp(year=1979, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1980, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1981, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1982, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1983, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1984, month=1, day=1)),
                        ("T1", pd.Timestamp(year=1985, month=1, day=1)),
                        ("T2", pd.Timestamp(year=1979, month=1, day=1)),
                        ("T2", pd.Timestamp(year=1980, month=1, day=1)),
                        ("T2", pd.Timestamp(year=1981, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1973, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1974, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1975, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1976, month=1, day=1)),
                        ("T3", pd.Timestamp(year=1977, month=1, day=1)),
                    ],
                    names=["series_name", "timestamp"],
                ),
                columns=["series_value"],
            ),
        ),
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader_no_start_timestamp.tsf",
            "default_tsf",
            pd.DataFrame(
                {
                    "series_name": ["T1", "T2", "T3"],
                    "series_value": [
                        [
                            25092.2284,
                            24271.5134,
                            25828.9883,
                            27697.5047,
                            27956.2276,
                            29924.4321,
                            30216.8321,
                        ],
                        [887896.51, 887068.98, 971549.04],
                        [227921, 230995, 183635, 238605, 254186],
                    ],
                }
            ),
        ),
        (
            "datasets/data/UnitTest/UnitTest_Tsf_Loader_no_start_timestamp.tsf",
            "pd-multiindex",
            pd.DataFrame(
                data=[
                    25092.2284,
                    24271.5134,
                    25828.9883,
                    27697.5047,
                    27956.2276,
                    29924.4321,
                    30216.8321,
                    887896.51,
                    887068.98,
                    971549.04,
                    227921,
                    230995,
                    183635,
                    238605,
                    254186,
                ],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("T1", 0),
                        ("T1", 1),
                        ("T1", 2),
                        ("T1", 3),
                        ("T1", 4),
                        ("T1", 5),
                        ("T1", 6),
                        ("T2", 0),
                        ("T2", 1),
                        ("T2", 2),
                        ("T3", 0),
                        ("T3", 1),
                        ("T3", 2),
                        ("T3", 3),
                        ("T3", 4),
                    ],
                    names=["series_name", "timestamp"],
                ),
                columns=["series_value"],
            ),
        ),
    ],
)
def test_load_tsf_to_dataframe(input_path, return_type, output_df):
    """Test function for loading tsf format."""
    data_path = os.path.join(
        os.path.dirname(sktime.__file__),
        input_path,
    )

    expected_metadata = {
        "frequency": "yearly",
        "forecast_horizon": 4,
        "contain_missing_values": False,
        "contain_equal_length": False,
    }

    df, metadata = load_tsf_to_dataframe(data_path, return_type=return_type)

    assert_frame_equal(df, output_df, check_dtype=False)
    assert metadata == expected_metadata
    if return_type != "default_tsf":
        assert check_is_mtype(obj=df, mtype=return_type)


@pytest.mark.parametrize("freq", [None, "YS"])
def test_convert_tsf_to_multiindex(freq):
    input_df = pd.DataFrame(
        {
            "series_name": ["T1", "T2", "T3"],
            "start_timestamp": [
                pd.Timestamp(year=1979, month=1, day=1),
                pd.Timestamp(year=1979, month=1, day=1),
                pd.Timestamp(year=1973, month=1, day=1),
            ],
            "series_value": [
                [
                    25092.2284,
                    24271.5134,
                    25828.9883,
                    27697.5047,
                    27956.2276,
                    29924.4321,
                    30216.8321,
                ],
                [887896.51, 887068.98, 971549.04],
                [227921, 230995, 183635, 238605, 254186],
            ],
        }
    )

    output_df = pd.DataFrame(
        data=[
            25092.2284,
            24271.5134,
            25828.9883,
            27697.5047,
            27956.2276,
            29924.4321,
            30216.8321,
            887896.51,
            887068.98,
            971549.04,
            227921,
            230995,
            183635,
            238605,
            254186,
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ("T1", pd.Timestamp(year=1979, month=1, day=1)),
                ("T1", pd.Timestamp(year=1980, month=1, day=1)),
                ("T1", pd.Timestamp(year=1981, month=1, day=1)),
                ("T1", pd.Timestamp(year=1982, month=1, day=1)),
                ("T1", pd.Timestamp(year=1983, month=1, day=1)),
                ("T1", pd.Timestamp(year=1984, month=1, day=1)),
                ("T1", pd.Timestamp(year=1985, month=1, day=1)),
                ("T2", pd.Timestamp(year=1979, month=1, day=1)),
                ("T2", pd.Timestamp(year=1980, month=1, day=1)),
                ("T2", pd.Timestamp(year=1981, month=1, day=1)),
                ("T3", pd.Timestamp(year=1973, month=1, day=1)),
                ("T3", pd.Timestamp(year=1974, month=1, day=1)),
                ("T3", pd.Timestamp(year=1975, month=1, day=1)),
                ("T3", pd.Timestamp(year=1976, month=1, day=1)),
                ("T3", pd.Timestamp(year=1977, month=1, day=1)),
            ],
            names=["series_name", "timestamp"],
        ),
        columns=["series_value"],
    )

    metadata = {
        "frequency": "yearly",
        "forecast_horizon": 4,
        "contain_missing_values": False,
        "contain_equal_length": False,
    }

    assert_frame_equal(
        output_df,
        _convert_tsf_to_hierarchical(input_df, metadata, freq=freq),
        check_dtype=False,
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("backoff", severity="none"),
    reason="load_solar requires backoff in the environment",
)
@pytest.mark.parametrize("return_df", [False, True])
def test_load_solar(return_df):
    """Test function for loading solar data through the Sheffiled Solar API."""
    # queried on 03/08/2022
    test_equals = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.002,
            0.021,
            0.06,
            0.113,
            0.176,
            0.251,
            0.323,
            0.356,
            0.395,
            0.427,
            0.431,
            0.398,
            0.396,
            0.406,
            0.413,
            0.43,
            0.421,
            0.414,
            0.391,
            0.37,
            0.315,
            0.258,
            0.222,
            0.203,
            0.174,
            0.134,
            0.093,
            0.055,
            0.022,
            0.002,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    y = load_solar(start="2021-05-01", end="2021-05-02", return_full_df=return_df)

    if return_df:
        assert isinstance(y, pd.DataFrame)
    else:
        assert isinstance(y, pd.Series)
        y = y.round(3).to_numpy()
        assert np.all(y == test_equals)
