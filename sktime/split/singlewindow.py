#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that produces a single train/test split based on a window."""

__author__ = ["khrapovs"]

__all__ = [
    "SingleWindowSplitter",
]

from typing import Optional

import numpy as np
import pandas as pd

from sktime.datatypes._utilities import get_index_for_series
from sktime.split.base import BaseSplitter
from sktime.split.base._common import (
    ACCEPTED_Y_TYPES,
    FORECASTING_HORIZON_TYPES,
    SPLIT_GENERATOR_TYPE,
    _check_fh,
    _check_inputs_for_compatibility,
    _get_end,
    _get_train_window_via_endpoint,
)
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    array_is_int,
    check_window_length,
)


class SingleWindowSplitter(BaseSplitter):
    r"""Single window splitter.

    Split time series once into a training and test set.
    See more details on what to expect from this splitter in :class:`BaseSplitter`.

    Test window is defined by forecasting horizons
    relative to the end of the training window.
    It will contain as many indices
    as there are forecasting horizons provided to the `fh` argument.
    For a forecasating horizon :math:`(h_1,\ldots,h_H)`, the training window will
    consist of the indices :math:`(k_n+h_1,\ldots,k_n+h_H)`.

    Parameters
    ----------
    fh : int, list or np.array
        Forecasting horizon
    window_length : int or timedelta or pd.DateOffset
        Window length

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import SingleWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = SingleWindowSplitter(fh=[2, 4], window_length=3)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([3, 4, 5]), array([7, 9]))]
    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES,
        window_length: Optional[ACCEPTED_WINDOW_LENGTH_TYPES] = None,
    ) -> None:
        _check_inputs_for_compatibility(args=[fh, window_length])
        super().__init__(fh=fh, window_length=window_length)

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        n_timepoints = y.shape[0]
        window_length = check_window_length(self.window_length, n_timepoints)
        fh = _check_fh(self.fh)
        train_end = _get_end(y_index=y, fh=fh)
        training_window = _get_train_window_via_endpoint(y, train_end, window_length)
        if array_is_int(fh):
            test_window = train_end + fh.to_numpy()
        else:
            test_window = y.get_indexer(y[train_end] + fh.to_pandas())

        yield training_window, test_window

    def get_n_splits(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> int:
        """Return the number of splits.

        Since this splitter returns a single train/test split,
        this number is trivially 1.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return 1

    def get_cutoffs(self, y: Optional[ACCEPTED_Y_TYPES] = None) -> np.ndarray:
        """Return the cutoff points in .iloc[] context.

        Since this splitter returns a single train/test split,
        this method returns a single one-dimensional array
        with the last train set index.

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
        fh = _check_fh(self.fh)
        y = get_index_for_series(y)
        end = _get_end(y_index=y, fh=fh)
        return np.array([end])

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
        return [{"fh": 3}, {"fh": [2, 4], "window_length": 3}]
