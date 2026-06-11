#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Forecasting horizon based train test split utility function."""

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.utils.validation.forecasting import check_fh


class ForecastingHorizonSplitter(BaseSplitter):
    r"""Splitter that creates a single train/test split based on a forecasting horizon.

    Handles both relative and absolute forecasting horizons.

    For a single series and a forecasting horizon correponsing to absolute
    time points :math:`(h_1, h_2, \ldots, h_H)`, splits into training and test set
    as follows:

    - training set: all time points strictly before :math:`h_1`
    - test set: all time points :math:`(h_1, h_2, \ldots, h_H)`

    For relative forecasting horizons, the last time point in the training set
    is assumed to be identical with :math:`h_H`, i.e., the last time point
    in the series to split.

    More precisely, if :math:`t_1, t_2, \ldots, t_N` are the time points in the series,
    and :math:`h_1, h_2, \ldots, h_H` are the relative forecasting horizons, i.e.,
    time offsets, then training and test sets are defined as follows:

    - training set: all time points :math:`t_i` such that :math:`t_i \lneq t_N - h_H`
    - test set: if :math:`t_j` is the last time point in the training set,
      then the test set consists of the time points
      :math:`(t_j + h_1, t_j + h_2, \ldots, t_j + h_H)`.

    Users should note that, for non-contiguous forecasting horizons,
    the union of training and test sets will not cover the entire time series.

    For zero or negative relative forecasting horizons, the training set
    will contain time points that are later than some time points in the test set,
    leading to leakage - users should ensure this is intentional when requested.

    Parameters
    ----------
    fh : ForecastingHorizon or compatible input
        Forecasting horizon that defines the test set.
        Must be all out-of-sample if relative.
    """

    _tags = {"split_hierarchical": False}

    def __init__(self, fh):
        super().__init__(fh=fh)

    def _split(self, y: pd.Index):
        """Return train/test indices based on forecasting horizon."""
        fh = check_fh(self.fh, freq=y)
        idx = fh.to_pandas()

        if fh.is_relative:
            min_step, max_step = idx.min(), idx.max()
            steps = fh.to_indexer()

            last_train_ix_minus_one = len(y) - max_step - 1
            first_test_ix = last_train_ix_minus_one + min(0, min_step - 1)

            train_ix = np.arange(last_train_ix_minus_one)
            test_ix = (np.arange(first_test_ix, len(y)))[steps]

        else:
            min_step, max_step = idx.min(), idx.max()

            train_ix = np.where(y < min_step)[0]
            test_ix = y.get_indexer(idx)

        yield train_ix, test_ix

    def get_n_splits(self, y: pd.Index | None = None) -> int:
        """Return number of splits (always 1)."""
        return 1

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter."""
        from sktime.forecasting.base import ForecastingHorizon

        fh_rel = ForecastingHorizon([1, 2, 3], is_relative=True)
        fh_abs = ForecastingHorizon([7, 8, 9], is_relative=True)
        fh_abs_2 = ForecastingHorizon([-2, 5], is_relative=True)

        # absolute horizons are tested indirectly through
        # testing temporal_train_test_split
        # it is not possible via get_test_params, since
        # the fh type must depend on the data index type

        return [{"fh": fh_rel}, {"fh": fh_abs}, {"fh": fh_abs_2}]
