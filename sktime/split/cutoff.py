#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter with a single train/test cutoff split."""

__author__ = ["khrapovs"]

__all__ = [
    "CutoffSplitter",
]

from typing import Optional

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.split.base._common import (
    ACCEPTED_Y_TYPES,
    DEFAULT_FH,
    DEFAULT_WINDOW_LENGTH,
    FORECASTING_HORIZON_TYPES,
    SPLIT_GENERATOR_TYPE,
    _check_fh,
    _check_inputs_for_compatibility,
    _get_train_window_via_endpoint,
)
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    array_is_datetime64,
    array_is_int,
    check_window_length,
    is_datetime,
    is_int,
    is_timedelta,
)
from sktime.utils.validation.forecasting import VALID_CUTOFF_TYPES, check_cutoffs


def _check_cutoffs_and_y(cutoffs: VALID_CUTOFF_TYPES, y: ACCEPTED_Y_TYPES) -> None:
    """Check that combination of inputs is compatible.

    Parameters
    ----------
    cutoffs : np.array or pd.Index
        cutoff points, positive and integer- or datetime-index like
    y : pd.Series, pd.DataFrame, np.ndarray, or pd.Index
        coerced and checked version of input y

    Raises
    ------
    ValueError
        if max cutoff is above the last observation in `y`
    TypeError
        if `cutoffs` type is not supported
    """
    max_cutoff = np.max(cutoffs)
    msg = (
        "`cutoffs` are incompatible with given `y`. "
        "Maximum cutoff is not smaller than the "
    )
    if array_is_int(cutoffs):
        if max_cutoff >= y.shape[0]:
            raise ValueError(msg + "number of observations.")
    elif array_is_datetime64(cutoffs):
        if max_cutoff >= np.max(y):
            raise ValueError(msg + "maximum index value of `y`.")
    else:
        raise TypeError("Unsupported type of `cutoffs`")


def _check_cutoffs_fh_y(
    cutoffs: VALID_CUTOFF_TYPES, fh: FORECASTING_HORIZON_TYPES, y: pd.Index
) -> None:
    """Check that combination of inputs is compatible.

    Currently, only two cases are allowed:
    either both `cutoffs` and `fh` are integers, or they are datetime or timedelta.

    Parameters
    ----------
    cutoffs : np.array or pd.Index
        Cutoff points, positive and integer- or datetime-index like.
        Type should match the type of `fh` input.
    fh : int, timedelta, list or np.ndarray of ints or timedeltas
        Type should match the type of `cutoffs` input.
    y : pd.Index
        Index of time series

    Raises
    ------
    ValueError
        if max cutoff plus max `fh` is above the last observation in `y`
    TypeError
        if `cutoffs` and `fh` type combination is not supported
    """
    max_cutoff = np.max(cutoffs)
    max_fh = fh.max()

    msg = "`fh` is incompatible with given `cutoffs` and `y`."
    if is_int(x=max_cutoff) and is_int(x=max_fh):
        if max_cutoff + max_fh > y.shape[0]:
            raise ValueError(msg)
    elif is_datetime(x=max_cutoff) and is_timedelta(x=max_fh):
        if max_cutoff + max_fh > y.max():
            raise ValueError(msg)
    else:
        raise TypeError("Unsupported type of `cutoffs` and `fh`")


class CutoffSplitter(BaseSplitter):
    r"""Cutoff window splitter.

    Split time series at given cutoff points into a fixed-length training and test set.

    Here the user is expected to provide a set of cutoffs (train set endpoints),
    which using the notation provided in :class:`BaseSplitter`,
    can be written as :math:`(k_1,\ldots,k_n)` for integer based indexing,
    or :math:`(t(k_1),\ldots,t(k_n))` for datetime based indexing.

    For a cutoff :math:`k_i` and a `window_length` :math:`w`
    the training window is :math:`(k_i-w+1,k_i-w+2,k_i-w+3,\ldots,k_i)`.
    Training window's last point is equal to the cutoff.

    Test window is defined by forecasting horizons
    relative to the end of the training window.
    It will contain as many indices
    as there are forecasting horizons provided to the `fh` argument.
    For a forecasating horizon :math:`(h_1,\ldots,h_H)`, the test window will
    consist of the indices :math:`(k_n+h_1,\ldots, k_n+h_H)`.

    The number of splits returned by `.get_n_splits`
    is then trivially equal to :math:`n`.

    The sorted array of cutoffs returned by `.get_cutoffs` is then equal to
    :math:`(t(k_1),\ldots,t(k_n))` with :math:`k_i<k_{i+1}`.

    Parameters
    ----------
    cutoffs : list or np.ndarray or pd.Index
        Cutoff points, positive and integer- or datetime-index like.
        Type should match the type of `fh` input.
    fh : int, timedelta, list or np.ndarray of ints or timedeltas
        Type should match the type of `cutoffs` input.
    window_length : int or timedelta or pd.DateOffset

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import CutoffSplitter
    >>> ts = np.arange(10)
    >>> splitter = CutoffSplitter(fh=[2, 4], cutoffs=np.array([3, 5]), window_length=3)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([1, 2, 3]), array([5, 7])), (array([3, 4, 5]), array([7, 9]))]
    """

    def __init__(
        self,
        cutoffs: VALID_CUTOFF_TYPES,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
    ) -> None:
        _check_inputs_for_compatibility([fh, cutoffs, window_length])
        self.cutoffs = cutoffs
        super().__init__(fh, window_length)

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        cutoffs = check_cutoffs(cutoffs=self.cutoffs)
        fh = _check_fh(fh=self.fh)
        window_length = check_window_length(
            window_length=self.window_length, n_timepoints=n_timepoints
        )
        _check_cutoffs_and_y(cutoffs=cutoffs, y=y)
        _check_cutoffs_fh_y(cutoffs=cutoffs, fh=fh, y=y)

        for cutoff in cutoffs:
            training_window = _get_train_window_via_endpoint(y, cutoff, window_length)
            test_window = cutoff + fh.to_numpy()
            if is_datetime(x=cutoff):
                test_window = y.get_indexer(test_window[test_window >= y.min()])
            yield training_window, test_window

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        For this splitter the number is trivially equal to
        the number of cutoffs given during instance initialization.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return len(self.cutoffs)

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.

        This method trivially returns the cutoffs given during instance initialization,
        in case these cutoffs are integer .iloc[] friendly indices.
        The only change is that the set of cutoffs is sorted from smallest to largest.
        When the given cutoffs are datetime-like,
        then this method returns corresponding integer indices.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        cutoffs : 1D np.ndarray of int
            iloc location indices, in reference to y, of cutoff indices
        """
        if array_is_int(self.cutoffs):
            return check_cutoffs(self.cutoffs)
        return np.argwhere(y.index.isin(check_cutoffs(self.cutoffs))).flatten()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [{"cutoffs": np.array([3, 7, 10])}, {"cutoffs": [21, 22]}]
