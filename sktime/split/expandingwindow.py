#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that successively expands the training window."""

__author__ = ["kkoralturk", "khrapovs"]

__all__ = [
    "ExpandingWindowSplitter",
]

from sktime.split.base import BaseWindowSplitter
from sktime.split.base._common import (
    DEFAULT_FH,
    DEFAULT_STEP_LENGTH,
    DEFAULT_WINDOW_LENGTH,
    FORECASTING_HORIZON_TYPES,
    SPLIT_GENERATOR_TYPE,
)
from sktime.utils.validation import (
    ACCEPTED_WINDOW_LENGTH_TYPES,
    NON_FLOAT_WINDOW_LENGTH_TYPES,
)


class ExpandingWindowSplitter(BaseWindowSplitter):
    r"""Expanding window splitter.

    Split time series repeatedly into a growing training set and a fixed-size test set.

    The training windows start at the first time index and expand by ``step_length``
    at each fold, starting with an initial window of size ``initial_window``.

    If the time points in the data are :math:`(t_1, t_2, \ldots, t_N)`, the training
    window for the :math:`n`-th split consists of all indices in the interval

    .. math:: [t_1,\; t_{W + (n-1) \cdot s}]

    where :math:`W` is the ``initial_window`` length and :math:`s` is the
    ``step_length``.

    The test windows are defined by forecasting horizons
    relative to the end of the training window.
    The test window will contain as many indices
    as there are forecasting horizons provided to the ``fh`` argument.

    For a forecasting horizon :math:`(h_1,\ldots,h_H)`, the test indices for the
    :math:`n`-th split will consist of the indices :math:`(k_n+h_1,\ldots,k_n+h_H)`,
    where :math:`k_n = t_{W + (n-1) \cdot s}` is the end of the :math:`n`-th
    training window.

    The number of splits is determined by the total length of the time series,
    up until the last test window that lies within the observed time indices,
    i.e., the largest integer :math:`n` such that :math:`k_n + h_H \leq t_N`.

    All indices are positional (``iloc``) indices, not label-based (``loc``),
    and apply to both regular and irregular time indices.

    For example for ``initial_window = 5``, ``step_length = 1`` and ``fh = [1, 2, 3]``
    here is a representation of the folds::

        |-----------------------|
        | * * * * * x x x - - - |
        | * * * * * * x x x - - |
        | * * * * * * * x x x - |
        | * * * * * * * * x x x |


    ``*`` = training fold.

    ``x`` = test fold.

    Parameters
    ----------
    fh : int, list or np.array, optional (default=1)
        Forecasting horizon, determines the test window. Should be relative.
        The test window is determined by applying the forecasting horizon ``fh``
        to the end of the training window.
    initial_window : int or timedelta or pd.DateOffset, optional (default=10)
        Window length of the initial (first) training fold.
        At each subsequent fold, the training window expands by ``step_length``.
        If ``initial_window = 0``, the initial training fold is empty.
    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between training windows, i.e., the amount by which
        the training window expands at each fold.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import ExpandingWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = ExpandingWindowSplitter(fh=[2, 4], initial_window=5, step_length=2)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    '[(array([0, 1, 2, 3, 4]), array([6, 8]))]'
    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        initial_window: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
    ) -> None:
        start_with_window = initial_window != 0

        # Note that we pass the initial window as the window_length below. This
        # allows us to use the common logic from the parent class, while at the same
        # time expose the more intuitive name for the ExpandingWindowSplitter.
        super().__init__(
            fh=fh,
            window_length=initial_window,
            initial_window=None,
            step_length=step_length,
            start_with_window=start_with_window,
        )

        # initial_window needs to be written to self for sklearn compatibility
        self.initial_window = initial_window
        # this class still acts as if it were overwritten with None,
        # via the _initial_window property that is read everywhere

    @property
    def _initial_window(self):
        return None

    def _split_windows(self, **kwargs) -> SPLIT_GENERATOR_TYPE:
        return self._split_windows_generic(expanding=True, **kwargs)

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
        return [{}, {"fh": [2, 4], "initial_window": 5, "step_length": 2}]
