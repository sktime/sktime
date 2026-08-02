#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Splitter that successively cuts test folds off the end of the series."""

__author__ = ["davidgilbertson"]

__all__ = [
    "ExpandingGreedySplitter",
]
__author__ = ["davidgilbertson"]


import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.split.base._common import SPLIT_GENERATOR_TYPE


class ExpandingGreedySplitter(BaseSplitter):
    r"""Splitter that successively cuts test folds off the end of the series.

    Takes an integer ``test_size`` that defines the number of steps included in the
    test set of each fold. The train set of each fold will contain all data before
    the test set. If the data contains multiple instances, ``test_size`` is
    _per instance_.

    If the time points in the data are :math:`(t_1, t_2, \ldots, t_N)`, with
    ``test_size`` :math:`= s` and ``folds`` :math:`= K`, the test windows are
    fixed-size windows of size :math:`s` cut from the end of the data backwards:

    .. math::

        [t_{N-Ks+1},\, t_{N-(K-1)s}], \quad
        [t_{N-(K-1)s+1},\, t_{N-(K-2)s}], \quad
        \ldots, \quad
        [t_{N-s+1},\, t_N]

    The corresponding training window for the :math:`n`-th fold expands from
    :math:`t_1` to cover all data before its test window:

    .. math::

        [t_1,\, t_{N-(K-n+1) \cdot s}] \quad \text{for } n = 1, 2, \ldots, K

    The number of splits is equal to ``folds``, i.e., :math:`K`.

    If no ``step_length`` is defined, the test sets (one for each fold) will be
    adjacent and disjoint, taken from the end of the dataset.

    For example, with ``test_size=7`` and ``folds=5``, the test sets in total will cover
    the last 35 steps of the data with no overlap.

    For example, for ``test_size=3`` and ``folds=2`` on a series of 10 steps,
    here is a representation of the folds::

        |---------------------|
        | * * * * x x x - - - |
        | * * * * * * * x x x |

    ``*`` = training fold.

    ``x`` = test fold.

    ``-`` = unused.

    Parameters
    ----------
    test_size : int or float
        If int: the number of steps included in the test set of each fold.
            Formally, steps are consecutive ``iloc`` indices.
        If float: the proportion of steps included in the test set of each fold,
            as a proportion of the total number of consecutive ``iloc`` indices.
            Must be between 0.0 and 1.0. Proportions are rounded to the
            next higher integer count of samples (ceil).
            Cave: not the ``loc`` proportion between start and end locations,
            but a proportion of total number of consecutive ``iloc`` indices.
    folds : int, default = 5
        The number of folds.
    step_length : int, optional
        The number of steps advanced for each fold. Defaults to ``test_size``.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.split import ExpandingGreedySplitter

    >>> ts = np.arange(10)
    >>> splitter = ExpandingGreedySplitter(test_size=3, folds=2)
    >>> list(splitter.split(ts))  # doctest: +SKIP
    [
        (array([0, 1, 2, 3]), array([4, 5, 6])),
        (array([0, 1, 2, 3, 4, 5, 6]), array([7, 8, 9]))
    ]
    """

    _tags = {"split_hierarchical": True}

    def __init__(
        self,
        test_size: int | float,
        folds: int = 5,
        step_length: int | None = None,
    ):
        super().__init__()
        self.test_size = test_size
        self.folds = folds
        self.step_length = step_length
        self.fh = np.arange(test_size) + 1 if isinstance(test_size, int) else None

        # no algorithm implemented that is faster for float than naive iteration
        if isinstance(test_size, float):
            self.set_tags(**{"split_hierarchical": False})

    def _split(self, y: pd.Index) -> SPLIT_GENERATOR_TYPE:
        test_size = self.test_size

        if isinstance(test_size, float):
            _test_size = int(np.ceil(len(y) * test_size))
            self.fh = np.arange(_test_size) + 1
        else:
            _test_size = test_size

        if isinstance(y, pd.MultiIndex):
            groups = pd.Series(index=y, dtype="float64").groupby(y.names[:-1])
            reverse_idx = groups.transform("size") - groups.cumcount() - 1
        else:
            reverse_idx = np.arange(len(y))[::-1]

        step_length = self.step_length or _test_size

        for i in reversed(range(self.folds)):
            tst_end = i * step_length
            trn_end = tst_end + _test_size
            trn_indices = np.flatnonzero(reverse_idx >= trn_end)
            tst_indices = np.flatnonzero(
                (reverse_idx < trn_end) & (reverse_idx >= tst_end)
            )
            yield trn_indices, tst_indices

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
        params1 = {"test_size": 1}
        params2 = {"test_size": 3, "folds": 2, "step_length": 2}
        params3 = {"test_size": 0.2, "folds": 2}
        return [params1, params2, params3]
