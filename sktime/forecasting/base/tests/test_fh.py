#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]

import numpy as np
import pytest
from sktime.forecasting import FH
from sktime.forecasting.tests import TEST_FHS
from sktime.utils.testing.forecasting import make_forecasting_problem
from sktime.utils.validation.forecasting import check_fh_values

TEST_CUTOFFS = [2, 5]
FH0 = TEST_FHS[0]


@pytest.mark.parametrize("fh", TEST_FHS)
def test_relative_relative(fh):
    fh = FH(fh, is_relative=True)
    relative = fh.relative()
    np.testing.assert_array_equal(relative, check_fh_values(fh))


def test_relative_absolute_no_cutoff():
    fh = FH(FH0, is_relative=True)
    with pytest.raises(ValueError):
        fh.absolute()


def test_absolute_relative_no_cutoff():
    fh = FH(FH0, is_relative=False)
    with pytest.raises(ValueError):
        fh.relative()
    with pytest.raises(ValueError):
        fh.in_sample()
    with pytest.raises(ValueError):
        fh.out_of_sample()


@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("cutoff", TEST_CUTOFFS)
def test_relative_in_and_out_of_sample(fh, cutoff):
    fh = FH(fh)
    ins = fh.in_sample(cutoff)
    assert all(ins <= 0)

    oos = fh.out_of_sample(cutoff)
    assert all(oos > 0)


def test_y_test_index_input():
    y_train, y_test = make_forecasting_problem()
    fh = FH(y_test.index, is_relative=False)
    cutoff = y_train.index[-1]
    np.testing.assert_array_equal(fh.relative(cutoff), np.arange(len(y_test)) + 1)
