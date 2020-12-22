# -*- coding: utf-8 -*-
__author__ = ["Jason Pong"]
__all__ = ["test_write_dataframe_to_ts_success", "test_write_dataframe_to_ts_fail"]

import os

import numpy as np
import pytest
from pandas._testing import assert_frame_equal

import sktime

from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_io import write_dataframe_to_tsfile


def test_write_dataframe_to_ts_success(tmp_path):
    # load an example dataset
    path = os.path.join(
        os.path.dirname(sktime.__file__),
        "datasets/data/ItalyPowerDemand/ItalyPowerDemand_TEST.ts",
    )
    test_x, test_y = load_from_tsfile_to_dataframe(path)
    # output the dataframe in a ts file
    write_dataframe_to_tsfile(
        data=test_x,
        path=tmp_path,
        problem_name="ItalyPowerDemand",
        timestamp=False,
        univariate=True,
        class_label=[1, 2],
        class_value_list=test_y,
        comment="""
          The data was derived from twelve monthly electrical power demand
          time series from Italy and first used in the paper "Intelligent
          Icons: Integrating Lite-Weight Data Mining and Visualization into
          GUI Operating Systems". The classification task is to distinguish
          days from Oct to March (inclusive) from April to September.
        """,
    )
    # load data back from the ts file
    result = f"{tmp_path}/ItalyPowerDemand/ItalyPowerDemand_transform.ts"
    res_x, res_y = load_from_tsfile_to_dataframe(result)
    # check if the dataframes are the same
    assert_frame_equal(res_x, test_x)


def test_write_dataframe_to_ts_fail(tmp_path):
    with pytest.raises(ValueError, match="Data provided must be a DataFrame"):
        write_dataframe_to_tsfile(
            data=np.random.rand(3, 2),
            path=str(tmp_path),
            problem_name="GunPoint",
            timestamp=False,
            univariate=True,
        )
