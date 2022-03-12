# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Series transformers for time series augmentation."""

__author__ = ["MrPr3ntice", "MFehsenfeld", "iljamaurer"]
__all__ = [
    "WhiteNoiseAugmenter",
    "ReverseAugmenter",
    "InvertAugmenter",
    "RandomSamplesAugmenter",
]


import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.utils import check_random_state

from sktime.transformations.base import BaseTransformer


class _AugmenterTags:
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "handles-missing-data": False,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "X-y-must-have-same-index": False,
        "fit-in-transform": True,
        "transform-returns-same-time-index": False,
        "capability:inverse_transform": False,
    }


class WhiteNoiseAugmenter(_AugmenterTags, BaseTransformer):
    r"""Augmenter adding Gaussian (i.e. white) noise to the time series.

    If `transform` is given time series $X={x_1, x_2, ... , x_n}$, then
    returns $X_t={x_1+e_1, x_2+e_2, ..., x_n+e_n}$ where $e_i$ are i.i.d. random draws from
    $\frac{1}{\sigma\sqrt{2 \pi}}e^{-\frac{1}{2}(\frac{x}{\sigma})}$,
    with $\sigma$ as the ``scale`` Factor.

    Parameters
    ----------
    scale: float, scale parameter (default=1.0)
            Specifies the standard deviation.
    random_state: None or int or ``np.random.RandomState`` instance, optional
            "If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None." [1]

    References and Footnotes
    ----------

        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.
            rv_continuous.random_state.html?highlight=random_state#scipy.stats.
            rv_continuous.random_state

    """

    _allowed_statistics = [np.std]

    def __init__(self, scale=1.0, random_state=42):
        self.scale = scale
        self.random_state = random_state
        super().__init__()

    def _transform(self, X, y=None):
        if self.scale in self._allowed_statistics:
            scale = self.scale(X)
        elif isinstance(self.scale, (int, float)):
            scale = self.scale
        else:
            raise TypeError(
                "Type of parameter 'scale' must be a float value or a distribution."
            )
        return X[0] + norm.rvs(0, scale, size=len(X), random_state=self.random_state)


class ReverseAugmenter(_AugmenterTags, BaseTransformer):
    r"""Augmenter reversing the time series.

    If `transform` is given a time series $X={x_1, x_2, ... , x_n}$, then
    returns $X_t={x_n, x_{n-1}, ..., x_2, x_1}.

    Example
    -------
    >>> X = pd.Series([1,2,3,4,5])
    >>> augmenter = ReverseAugmenter()
    >>> Xt = augmenter.fit_transform(X)
    >>> Xt
    0    5
    1    4
    2    3
    3    2
    4    1
    dtype: int64
    """

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X.loc[::-1].reset_index(drop=True, inplace=False)


class InvertAugmenter(_AugmenterTags, BaseTransformer):
    r"""Augmenter inverting the time series by multiplying it by -1.

    If given time series $X={x_1, x_2, ... , x_n}$, then
    $X_t={-x_1, -x_2, ... , -x_n}.

    Example
    -------
    >>> X = pd.Series([1,2,3,4,5])
    >>> augmenter = InvertAugmenter()
    >>> Xt = augmenter.fit_transform(X)
    >>> Xt
    0   -1
    1   -2
    2   -3
    3   -4
    4   -5
    dtype: int64
    """

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X.mul(-1)


class RandomSamplesAugmenter(_AugmenterTags, BaseTransformer):
    r"""Draw random samples form time series.

    If given time series $X \in \mathbb{R}^n$ then $X_t \subseteq X$
    randomly drawn from $X$ (with or without replacement).

    Parameters
    ----------
    n: int or float, optional (default = 1.0)
            To specify an exact number of samples to draw, set `n` to an int value.
            Number of samples to draw. If type of `n` is float,
            To specify the returned samples as a proportion of the given times series
            set `n` to an float value.
            By default, the same number of samples is returned as in the given times
            series
    without_replacement: bool, optional (default = True)
            Whether to draw without replacement. If True, every samples of given
            times series `X` appears once in `Xt`.
    random_state: None or int or ``np.random.RandomState`` instance, optional
            "If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None." [1]

    References and Footnotes
    ----------

        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.
        rv_continuous.random_state.html?highlight=random_state#scipy.
        stats.rv_continuous.random_state

    """

    def __init__(
        self,
        n=1.0,
        without_replacement=True,
        random_state=42,
    ):
        if isinstance(n, float):
            if n <= 0.0 or not np.isfinite(n):
                raise ValueError("n must be a positive, finite number.")
        elif isinstance(n, int):
            if n < 1 or not np.isfinite(n):
                raise ValueError("n must be a finite number >= 1.")
        else:
            raise ValueError("n must be int or float, not " + str(type(n))) + "."
        self.n = n
        self.without_replacement = without_replacement
        self.random_state = random_state
        super().__init__()

    def _transform(self, X, y=None):
        if isinstance(self.n, float):
            n = int(np.ceil(self.n * len(X)))
        else:
            n = self.n
        rng = check_random_state(self.random_state)
        values = np.concatenate(X.values)
        if self.without_replacement:
            replace = False
        else:
            replace = True
        Xt = rng.choice(values, n, replace)
        return pd.DataFrame(Xt)
