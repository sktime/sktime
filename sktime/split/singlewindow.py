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

    Split time series into a single training and test set window.

    The training set is defined based on a "window".
    The endpoint of the training set is determined by
    `the length of the time series - fh[-1] - 1`,
    and the starting point is calculated as `endpoint - window_length + 1`.
    If the starting point is negative, it will be set to 0.

    If the time points in the data are :math:`(t_1, t_2, \ldots, t_N)`,
    the training windows will be all indices in the interval

    .. math:: [t_N-fh[-1] - w, \ldots, t_N-fh[-1] - 1]

    where :math:`w` is the window length and N the length of the time series.

    The test window will contain as many indices
    as there are forecasting horizons provided to the ``fh`` argument.
    In particularly, they will be equal to the endpoint plus the forecasting horizon.

    For a forecasting horizon :math:`(h_1,\ldots,h_H)`, the test indices
    will consist of the indices :math:`(k+h_1,\ldots,k+h_H)`,
    where k is the end of the training window.


    **Important Notes:**

    - ``SingleWindowSplitter`` uses positional indexing (`iloc`) for the training and
        test windows, regardless of the type of ``window_length``.
        Even if ``window_length`` is a `timedelta` or `pd.DateOffset`,
        the splitter interprets it in terms of the number of positions.

    - ``window_length`` can be an integer, `timedelta`, or `pd.DateOffset`, where:

      - If `int`, it specifies the number of time points directly.

      - If `timedelta` or `pd.DateOffset`, it represents a relative duration,
        but it will still be applied as a positional offset,
        not based on label-based indexing.


    **Example Calculation:**
    For example, with ``window_length = 5``, ``fh = [1, 2, 3]`` and time points
    :math:`(t_0, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_{10})`,
    the resulting folds are as follows:

    - `[3, 4, 5, 6, 7]` = training fold indices.

    - `[8, 9, 10]` = test fold indices.

    Parameters
    ----------
    fh : int, list or np.array, optional (default=1)
        Forecasting horizon, determines the test window. Should be relative.
        The test window is determined by applying the forecasting horizon ``fh``
        to the end of the training window.

    window_length : int or timedelta or pd.DateOffset, optional (default=10)
        Window length of the training window.

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
        return [{"fh": 3}, {"fh": [2, 4], "window_length": 3}]
