#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Forecasting horizon based train test split utility function."""

from typing import Optional

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.utils.validation.forecasting import check_fh


class ForecastingHorizonSplitter(BaseSplitter):
    """Splitter that creates a single train/test split based on a forecasting horizon.

    Handles both relative and absolute forecasting horizons.

    Parameters
    ----------
    fh : ForecastingHorizon or compatible input
        Forecasting horizon that defines the test set.
        Must be all out-of-sample if relative.
    """

    _tags = {"split_hierarchical": False}

    def __init__(self, fh):
        self.fh = fh
        super().__init__()

    def _split(self, y: pd.Index):
        """Return train/test indices based on forecasting horizon."""
        ix = y.index
        fh = check_fh(self.fh, freq=ix)
        idx = fh.to_pandas()

        if fh.is_relative:
            if not fh.is_all_out_of_sample():
                raise ValueError("`fh` must only contain out-of-sample values")

            max_step = idx.max()
            steps = fh.to_indexer()

            train_ix = np.arange(len(y) - max_step)
            test_ix = (np.arange(len(y) - max_step, len(y)))[steps]

        else:
            min_step, max_step = idx.min(), idx.max()

            train_ix = np.where(ix < min_step)[0]
            test_ix = np.where((ix >= min_step) & (ix <= max_step))[0]

        yield train_ix, test_ix

    def get_n_splits(self, y: Optional[pd.Index] = None) -> int:
        """Return number of splits (always 1)."""
        return 1

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter."""
        from sktime.forecasting.base import ForecastingHorizon

        fh_rel = ForecastingHorizon([1, 2, 3], is_relative=True)
        fh_abs = ForecastingHorizon([7, 8, 9], is_relative=False)

        return [{"fh": fh_rel}, {"fh": fh_abs}]
