#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that combines expanding and sliding window strategies."""

__author__ = ["markussagen"]

__all__ = [
    "ExpandingSlidingWindowSplitter",
]

from sktime.split.base import BaseWindowSplitter


class ExpandingSlidingWindowSplitter(BaseWindowSplitter):
    r"""Combined expanding and sliding window splitter.

    Split time series repeatedly into a training window and a test window.
    The training window first grows (expanding window phase), until it reaches
    ``max_expanding_window_length`` time points. From then on, it keeps this length
    and moves forward (sliding window phase).

    If the time points in the data are :math:`(t_1, t_2, \ldots, t_N)`,
    the :math:`n`-th training window ends at the time point :math:`t_{k_n}`, where

    .. math:: k_n = w + (n - 1) \cdot s,

    with :math:`w` the initial window length ``initial_window``,
    and :math:`s` the step length ``step_length``.

    The :math:`n`-th training window consists of all time points :math:`t_i`
    with :math:`\max(1, k_n - L + 1) \leq i \leq k_n`,
    where :math:`L` is ``max_expanding_window_length``. That is:

    - while :math:`k_n \leq L` (expanding phase), the training window consists of
      the first :math:`k_n` time points, :math:`t_1, \ldots, t_{k_n}`;
    - once :math:`k_n > L` (sliding phase), the training window consists of
      the :math:`L` time points :math:`t_{k_n - L + 1}, \ldots, t_{k_n}`.

    If ``initial_window`` is larger than ``max_expanding_window_length``,
    there is no expanding phase, and all training windows have length :math:`L`.

    The test windows are defined by the forecasting horizon ``fh``,
    relative to the end of the training windows.
    For a forecasting horizon :math:`(h_1, \ldots, h_H)`, the test window of the
    :math:`n`-th split consists of the time points
    :math:`(t_{k_n + h_1}, \ldots, t_{k_n + h_H})`.

    The number of splits is the largest integer :math:`n` such that
    :math:`k_n + h_H \leq N`, i.e., such that the last test window still lies
    within the observed time points.
    If not even the first split fits, i.e., if :math:`w + h_H > N`,
    a ``ValueError`` is raised.

    The above describes integer ``initial_window`` and ``step_length``,
    where lengths are counted in time points (``iloc`` positions).

    For example, for a series with 10 time points, ``initial_window = 1``,
    ``step_length = 1``, ``fh = [1, 2]``, and ``max_expanding_window_length = 5``,
    here is a representation of the folds::

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

    ``*`` = training fold.

    ``x`` = test fold.

    ``-`` = unused observations.

    Parameters
    ----------
    fh : int, list or np.array, optional (default=1)
        Forecasting horizon, determines the test window. Should be relative.
        The test window is determined by applying the forecasting horizon ``fh``
        to the end of the training window.

    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between the ends of consecutive training windows.

    initial_window : int or timedelta or pd.DateOffset, optional (default=10)
        Length of the first training window, at the start of the expanding phase.

    max_expanding_window_length : int, optional (default=float("inf"))
        Maximum length of the training window, in number of time points.
        Once the expanding training window reaches this length, the splitter
        switches to a sliding training window of this length.
        If not set, the training window expands indefinitely,
        as in ``ExpandingWindowSplitter``.

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
    [(array([0]), array([1, 2])), (array([0, 1, 2, 3]), array([4, 5])),
      (array([2, 3, 4, 5, 6]), array([7, 8]))]

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
