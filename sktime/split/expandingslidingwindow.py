#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that combines expanding and sliding window strategies."""

__author__ = ["markussagen"]

__all__ = [
    "ExpandingSlidingWindowSplitter",
]

from sktime.split.base import BaseWindowSplitter


class ExpandingSlidingWindowSplitter(BaseWindowSplitter):
    r"""Combined Expanding and Sliding Window Splitter.

    This splitter starts as an expanding window splitter until a specified
    maximum window length is reached, then transitions to a sliding window splitter.

    For example, with ``initial_window = 1``, ``step_length = 1``,
    ``fh = [1, 2]``,, and ``max_expanding_window_length = 8``

    |--------------------|
    | * x x - - - - - - -| Expanding (initial_window = 1)
    | * * x x - - - - - -| Expanding
    | * * * x x - - - - -| Expanding
    | * * * * x x - - - -| Expanding
    | * * * * * x x - - -| Expanding
    | - * * * * * x x - -| Sliding (switched)
    | - - * * * * * x x -| Sliding
    | - - - * * * * * x x| Sliding
    |--------------------|
      0 1 2 3 4 5 6 7 8 9
                ^
                |_ maximum expanding window size reached

    ``*`` = training fold
    ``x`` = test fold
    ``-`` = unused observations

    Parameters
    ----------
    fh : int, list or np.array, optional (default=1)
        Forecasting horizon
    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between windows
    initial_window : int or timedelta or pd.DateOffset, optional (default=10)
        Initial window length for the expanding window phase
    max_expanding_window_length : int, optional (default=float('inf'))
        Maximum window length. If none is passed in, it will expanding indefinitely.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import ExpandingSlidingWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = ExpandingSlidingWindowSplitter(
    ...     fh=[1, 2],
    ...     step_length=3,
    ...     initial_window=1,
    ...     max_expanding_window_length=5,
    ... )
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([0]), array([1, 2])), (array([0, 1]), array([2, 3])),
      (array([0, 1, 2]), array([3, 4])), (array([0, 1, 2, 3]), array([4, 5])),
      (array([0, 1, 2, 3, 4]), array([5, 6])), (array([1, 2, 3, 4, 5]), array([6, 7])),
      (array([2, 3, 4, 5, 6]), array([7, 8])), (array([3, 4, 5, 6, 7]), array([8, 9]))]

    >>> import numpy as np
    >>> from sktime.split import ExpandingSlidingWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = ExpandingSlidingWindowSplitter(
    ...     fh=[1, 2],
    ...     step_length=3,
    ...     initial_window=2,
    ...     max_expanding_window_length=5,
    ... )
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([0, 1]), array([2, 3])), (array([0, 1, 2, 3, 4]), array([5, 6])),
      (array([3, 4, 5, 6, 7]), array([8, 9]))]
    """

    def __init__(
        self,
        fh=1,
        step_length=1,
        initial_window=10,
        max_expanding_window_length=float("inf"),
    ):
        start_with_window = initial_window != 0
        self.fh = fh

        super().__init__(
            window_length=initial_window,
            initial_window=None,
            step_length=step_length,
            start_with_window=start_with_window,
        )

        # initial_window needs to be written to self for sklearn compatibility
        self.initial_window = initial_window
        # this class still acts as if it were overwritten with None,
        # via the _initial_window property that is read everywhere

        # Defines the maximum length of the expanding window
        #  before switching to a sliding window splitter strategy
        self.max_expanding_window_length = max_expanding_window_length

    @property
    def _initial_window(self):
        return None

    def _split_windows(self, **kwargs):
        return self._split_windows_generic(expanding=True, **kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the splitter."""
        params1 = {
            "fh": [1, 2],
            "step_length": 1,
            "initial_window": 1,
            "max_expanding_window_length": 5,
        }
        params2 = {
            "fh": [1, 2],
            "step_length": 3,
            "initial_window": 2,
            "max_expanding_window_length": 5,
        }

        return [params1, params2]
