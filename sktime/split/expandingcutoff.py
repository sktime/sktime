#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that successively expands the training window."""

__author__ = ["ninedigits"]

__all__ = [
    "ExpandingCutoffSplitter",
]

from typing import Optional

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.split.base._common import ACCEPTED_Y_TYPES, _check_fh, _inputs_are_supported
from sktime.utils.validation.forecasting import check_step_length


def _is_all_args_periodlike(args):
    for arg in args:
        if not isinstance(arg, pd.Period):
            return False
    return True


class ExpandingCutoffSplitter(BaseSplitter):
    """
    Expanding cutoff splitter for time series data.

    This splitter combines elements of `ExpandingWindowSplitter` and `CutoffSplitter`
    to create training and testing sets. Unlike `ExpandingWindowSplitter` which begins
    with a fixed initial window, this splitter uses a specific cutoff point as the
    starting window for the training set. The training set then expands incrementally
    in each split until it reaches the end of the series.

    The test set is defined by a forecast horizon relative to the last point in the
    training set, containing as many subsequent indices as specified by the `fh`
    parameter.
                          c
                          ↓
    |---------------------|←---fh---→|------|
    | * * * * * * * * * * x x x x x x - - - |
    | * * * * * * * * * * * x x x x x x - - |
    | * * * * * * * * * * * * x x x x x x - |
    | * * * * * * * * * * * * * x x x x x x |

    ``c`` = cutoff date or index.

    ``*`` = training fold.

    ``x`` = test fold.

    Parameters
    ----------
        cutoff (int or pd.Timestamp):
            The initial cutoff point in the series, which marks the beginning of the
            first test set.
        fh (int, list, or np.array):
            Forecasting horizon, determining the size and  indices of the test sets.
            It can be an integer, a list, or an array.
        step_length (int):
            The step length to expand the training set size in each split.
    """

    def __init__(self, cutoff, fh, step_length):
        super().__init__()
        self.cutoff = cutoff
        self.fh = _check_fh(fh)
        self.step_length = step_length
        return

    def _split(self, y, fh=None):
        """
        Generate indices to split data into training and testing sets.

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            Time series data.
        fh : int, default=None
            Forecast horizon, if None, uses self.fh

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if fh is None:
            fh = self.fh
        offset = fh.to_numpy().max() + 1
        for cutoff in self.get_cutoffs(y):
            train_window = np.arange(0, cutoff + 1, step=self.step_length)
            test_window = np.arange(cutoff + 1, cutoff + offset, step=self.step_length)
            yield train_window, test_window

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : 1D np.ndarray of int
            iloc location indices, in reference to y, of cutoff indices
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `y` to compute the cutoffs."
            )
        y = self._coerce_to_index(y)
        input_args = [y[0], self.cutoff]
        if not _inputs_are_supported(input_args) and not _is_all_args_periodlike(
            input_args
        ):
            msg = (
                "y indicies and cutoff must have the same datatypes, but instead "
                f"found type(y) = {y.dtype} and type(cutoff) = {type(self.cutoff)}"
            )
            raise TypeError(msg)

        fh = self.fh
        step_length = check_step_length(self.step_length)
        cutoff_index = np.argmax(y == self.cutoff) - 1
        cutoffs = np.array([cutoff_index])
        offset = fh.to_numpy().max()
        while cutoff_index + offset < len(y) - 1:
            cutoff_index += step_length
            cutoffs = np.append(cutoffs, cutoff_index)
        return cutoffs

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return [{"fh": [2, 4], "cutoff": 3, "step_length": 1}]
