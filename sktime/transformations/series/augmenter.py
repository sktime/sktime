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
        "fit_is_empty": True,
        "transform-returns-same-time-index": False,
        "capability:inverse_transform": False,
    }


class WhiteNoiseAugmenter(_AugmenterTags, BaseTransformer):
    r"""Augmenter adding Gaussian (i.e. white) noise to the time series.

    If `transform` is given time series :math:`X={x_1, x_2, ... , x_n}`, then
    returns :math:`X_t={x_1+e_1, x_2+e_2, ..., x_n+e_n}` where :math:`e_i` are
    i.i.d. random draws from a normal distribution with mean :math:`\mu` = 0
    and standard deviation :math:`\sigma` = ``scale``.
    Time series augmentation by adding Gaussian Noise has been discussed among
    others in [1] and [2].

    Parameters
    ----------
    scale: float, scale parameter (default=1.0)
            Specifies the standard deviation.
    random_state: None or int or ``np.random.RandomState`` instance, optional
            "If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None." [3]

    References and Footnotes
    ----------

        [1]: WEN, Qingsong, et al. Time series data augmentation for deep
        learning: A survey. arXiv preprint arXiv:2002.12478, 2020.
        [2]: IWANA, Brian Kenji; UCHIDA, Seiichi. An empirical survey of data
        augmentation for time series classification with neural networks. Plos
        one, 2021, 16. Jg., Nr. 7, S. e0254841.
        [3]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.random_state.html # noqa

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
                "Type of parameter 'scale' must be a non-negative float value."
            )
        return X[0] + norm.rvs(0, scale, size=len(X), random_state=self.random_state)


class ReverseAugmenter(_AugmenterTags, BaseTransformer):
    r"""Augmenter reversing the time series.

    If `transform` is given a time series :math:`X={x_1, x_2, ... , x_n}`, then
    returns :math:`X_t={x_n, x_{n-1}, ..., x_2, x_1}`.
    Time series augmentation by reversing has been discussed e.g. in [1].

    Examples
    --------
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

    References and Footnotes
    ----------

        [1]: IWANA, Brian Kenji; UCHIDA, Seiichi. An empirical survey of data
        augmentation for time series classification with neural networks. Plos
        one, 2021, 16. Jg., Nr. 7, S. e0254841.

    """

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X.loc[::-1].reset_index(drop=True, inplace=False)


class InvertAugmenter(_AugmenterTags, BaseTransformer):
    r"""Augmenter inverting the time series by multiplying it by -1.

    If `transform` is given a time series :math:`X={x_1, x_2, ... , x_n}`, then
    returns :math:`X_t={-x_1, -x_2, ... , -x_n}`.

    Examples
    --------
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
    r"""Draw random samples from time series.

    `transform` takes a time series :math:`X={x_1, x_2, ... , x_m}` with :math:`m`
    elements and returns :math:`X_t={x_i, x_{i+1}, ... , x_n}`, where
    :math:`{x_i, x_{i+1}, ... , x_n}` are :math:`n`=``n`` random samples drawn
    from :math:`X` (with or `without_replacement`).

    Parameters
    ----------
    n: int or float, optional (default = 1.0)
            To specify an exact number of samples to draw, set `n` to an int value.
            Number of samples to draw.
            To specify the returned samples as a proportion of the given times series
            set `n` to a float value :math:`n \in [0, 1]`.
            By default, the same number of samples is returned as given by the input
            time series.
    without_replacement: bool, optional (default = True)
            Whether to draw without replacement. If True, every sample of the input
            times series `X` will appear at most once in `Xt`.
    random_state: None or int or ``np.random.RandomState`` instance, optional
            "If int or RandomState, use it for drawing the random variates.
            If None, rely on ``self.random_state``.
            Default is None." [1]

    References and Footnotes
    ----------

        [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.random_state.html # noqa

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
