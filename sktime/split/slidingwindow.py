#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that slides a fixed size window to generate train and test folds."""

__author__ = ["khrapovs", "mloning"]

__all__ = [
    "SlidingWindowSplitter",
]

from typing import Optional

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


class SlidingWindowSplitter(BaseWindowSplitter):
    r"""Sliding window splitter.

    Split time series repeatedly into a fixed-length training and test set.

    Test window is defined by forecasting horizons
    relative to the end of the training window.
    It will contain as many indices
    as there are forecasting horizons provided to the `fh` argument.
    For a forecasating horizon :math:`(h_1,\ldots,h_H)`, the training window will
    consist of the indices :math:`(k_n+h_1,\ldots,k_n+h_H)`.

    For example for `window_length = 5`, `step_length = 1` and `fh = [1, 2, 3]`
    here is a representation of the folds::

    |-----------------------|
    | * * * * * x x x - - - |
    | - * * * * * x x x - - |
    | - - * * * * * x x x - |
    | - - - * * * * * x x x |

    ``*`` = training fold.

    ``x`` = test fold.

    Parameters
    ----------
    fh : int, list or np.array
        Forecasting horizon
    window_length : int or timedelta or pd.DateOffset
        Window length
    step_length : int or timedelta or pd.DateOffset, optional (default=1)
        Step length between windows
    initial_window : int or timedelta or pd.DateOffset, optional (default=None)
        Window length of first window
    start_with_window : bool, optional (default=True)
        - If True, starts with full window.
        - If False, starts with empty window.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import SlidingWindowSplitter
    >>> ts = np.arange(10)
    >>> splitter = SlidingWindowSplitter(fh=[2, 4], window_length=3, step_length=2)
    >>> list(splitter.split(ts)) # doctest: +SKIP
    [(array([0, 1, 2]), array([4, 6])), (array([2, 3, 4]), array([6, 8]))]
    """

    def __init__(
        self,
        fh: FORECASTING_HORIZON_TYPES = DEFAULT_FH,
        window_length: ACCEPTED_WINDOW_LENGTH_TYPES = DEFAULT_WINDOW_LENGTH,
        step_length: NON_FLOAT_WINDOW_LENGTH_TYPES = DEFAULT_STEP_LENGTH,
        initial_window: Optional[ACCEPTED_WINDOW_LENGTH_TYPES] = None,
        start_with_window: bool = True,
    ) -> None:
        super().__init__(
            fh=fh,
            window_length=window_length,
            initial_window=initial_window,
            step_length=step_length,
            start_with_window=start_with_window,
        )

    def _split_windows(self, **kwargs) -> SPLIT_GENERATOR_TYPE:
        return self._split_windows_generic(expanding=False, **kwargs)

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
        return [{}, {"fh": [2, 4], "window_length": 3, "step_length": 2}]
