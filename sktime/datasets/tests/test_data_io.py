# -*- coding: utf-8 -*-
"""Test functions for data input and output."""

__author__ = [
    "SebasKoel",
    "Emiliathewolf",
    "TonyBagnall",
    "jasonlines",
]

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
    load_from_long_to_dataframe,
    load_from_tsfile,
    load_from_tsfile_to_dataframe,
    load_tsf_to_dataframe,
    load_uschange,
    write_dataframe_to_tsfile,
)
from sktime.datasets._data_io import MODULE


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


def test_load_tsf_to_dataframe():
    """Test function for loading tsf format."""
    data_path = os.path.join(
        os.path.dirname(sktime.__file__),
        "datasets/data/UnitTest/UnitTest_Tsf_Loader.tsf",
    )

    df, frequency, horizon, missing_values, equal_length = load_tsf_to_dataframe(
        data_path
    )

    test_df = pd.DataFrame(
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

    assert_frame_equal(df, test_df)
    assert frequency == "yearly"
    assert horizon == 4
    assert missing_values is False
    assert equal_length is False
