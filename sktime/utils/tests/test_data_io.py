# -*- coding: utf-8 -*-
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from sktime.utils.data_io import TsFileParseException
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_long_to_dataframe
from sktime.utils.data_io import LongFormatDataParseException
from sktime.utils.data_io import generate_example_long_table


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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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

            np.testing.assert_raises(
                TsFileParseException, load_from_tsfile_to_dataframe, path
            )

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
    # create and save a example long-format file to csv
    test_dataframe = generate_example_long_table()
    dataframe_path = tmpdir.join("data.csv")
    test_dataframe.to_csv(dataframe_path, index=False)
    # load and convert the csv to sktime-formatted data
    nested_dataframe = load_from_long_to_dataframe(dataframe_path)
    assert isinstance(nested_dataframe, pd.DataFrame)


def test_load_from_long_incorrect_format(tmpdir):
    with pytest.raises(LongFormatDataParseException):
        dataframe = generate_example_long_table()
        dataframe.drop(dataframe.columns[[3]], axis=1, inplace=True)
        dataframe_path = tmpdir.join("data.csv")
        dataframe.to_csv(dataframe_path, index=False)
        load_from_long_to_dataframe(dataframe_path)
