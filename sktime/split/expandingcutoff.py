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
from sktime.split.base._common import ACCEPTED_Y_TYPES, _check_fh
from sktime.utils.validation.forecasting import check_step_length


def _is_datetimelike(x):
    return isinstance(x, (pd.Timestamp, np.datetime64, pd.Period))


def _is_int(arg):
    return isinstance(arg, (int, np.int_))


def _validate_cutoff(cutoff):
    if not (_is_datetimelike(cutoff) or _is_int(cutoff)):
        raise TypeError(
            "Cutoff value must be datetime-like or integer but instead "
            f"found type(cutoff) = {type(cutoff)}"
        )
    return cutoff


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

    The valid types of y-index and cutoff pairings are datelike-datelike, datelike-int,
    and int-int. When a datelike index is combined with an int cutoff, the cutoff
    functions as an iloc indexer. When an int index is paired with a positive int
    cutoff, the cutoff serves as a loc indexer. If the int cutoff is negative, it
    functions as an iloc indexer.

    For example for ``cutoff = 10``, ``step_length = 1`` and ``fh = [1, 2, 3, 4, 5, 6]``
    here is a representation of the folds:

    ```
                          c
    |---------------------|----fh----|------|
    | * * * * * * * * * * x x x x x x - - - |
    | * * * * * * * * * * * x x x x x x - - |
    | * * * * * * * * * * * * x x x x x x - |
    | * * * * * * * * * * * * * x x x x x x |

    ```

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

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.split import ExpandingCutoffSplitter
    >>> date_range = pd.date_range(start='2020-Q1', end='2021-Q3', freq='QS')
    >>> y = pd.DataFrame(index=pd.PeriodIndex(date_range, freq='Q'))
    >>> cutoff = pd.Period('2021-Q1')
    >>> cv = ExpandingCutoffSplitter(cutoff=cutoff, fh=[1, 2], step_length=1)
    >>> list(cv.split(y)) # doctest: +SKIP
    [(array([0, 1, 2, 3]), array([4, 5])), (array([0, 1, 2, 3, 4]), array([5, 6]))]
    """

    def __init__(self, cutoff, fh, step_length):
        super().__init__()
        self.cutoff = _validate_cutoff(cutoff)
        self.fh = fh
        self._fh = _check_fh(fh)
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
            fh = self._fh
        for cutoff in self.get_cutoffs(y):
            train_window = np.arange(0, cutoff + 1, step=1)
            test_window = cutoff + fh
            yield train_window, test_window

    def _validate_y(self, y):
        y = self._coerce_to_index(y)
        cutoff = self.cutoff
        y0 = y[0]
        cond1 = _is_datetimelike(y0) and (_is_datetimelike(cutoff) or _is_int(cutoff))
        cond2 = _is_int(y0) and _is_int(cutoff)
        if not (cond1 or cond2):
            raise TypeError(
                "Valid combinations for y and cutoff types are "
                "datelike-datelike, datelike-int, or int-int, but instead "
                f"found {type(y0)}-{type(cutoff)}"
            )
        return y

    def _get_first_cutoff_index(self, y_index):
        y0 = y_index[0]
        if _is_int(self.cutoff) and self.cutoff < 0:
            index = len(y_index) + self.cutoff - 1
            if index < 0:
                raise IndexError(
                    f"Cutoff of value {self.cutoff} is out of bounds "
                    f"for y of length {len(y_index)}"
                )
        elif _is_datetimelike(y0) and _is_int(self.cutoff) and (self.cutoff > 0):
            index = self.cutoff - 1
        else:
            index = np.argmax(y_index == self.cutoff) - 1
            if index == -1:
                raise TypeError(
                    "Could not find matching index, make sure that "
                    f"type(y) {type(y0)} is compatible with type(cutoff) "
                    f"{type(self.cutoff)}"
                )
        return index

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
        y = self._validate_y(y)
        fh = self._fh
        step_length = check_step_length(self.step_length)
        cutoff_index = self._get_first_cutoff_index(y)
        cutoffs = np.array([cutoff_index])
        offset = fh.to_numpy().max()
        while cutoff_index + offset + step_length < len(y):
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
        return [
            {"fh": [2, 4], "cutoff": 3, "step_length": 1},
            {"fh": [1, 2, 3], "cutoff": 3, "step_length": 2},
        ]
