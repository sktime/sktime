# -*- coding: utf-8 -*-
"""Move with the restructure over to datasets."""

import pandas as pd
import pytest

from sktime.utils.data_io import (
    LongFormatDataParseException,
    generate_example_long_table,
    load_from_long_to_dataframe,
)


def test_load_from_long_to_dataframe(tmpdir):
    """Test loading long to dataframe."""
    # create and save a example long-format file to csv
    test_dataframe = generate_example_long_table()
    dataframe_path = tmpdir.join("data.csv")
    test_dataframe.to_csv(dataframe_path, index=False)
    # load and convert the csv to sktime-formatted data
    nested_dataframe = load_from_long_to_dataframe(dataframe_path)
    assert isinstance(nested_dataframe, pd.DataFrame)


def test_load_from_long_incorrect_format(tmpdir):
    """Test loading incorrect long to dataframe."""
    with pytest.raises(LongFormatDataParseException):
        dataframe = generate_example_long_table()
        dataframe.drop(dataframe.columns[[3]], axis=1, inplace=True)
        dataframe_path = tmpdir.join("data.csv")
        dataframe.to_csv(dataframe_path, index=False)
        load_from_long_to_dataframe(dataframe_path)
