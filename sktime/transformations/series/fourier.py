# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Fourier features."""

__author__ = ["mloning"]

import numpy as np
import pandas as pd
from pmdarima.preprocessing import FourierFeaturizer as _FourierFeaturizer

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.forecasting import check_sp


class FourierFeatures(BaseTransformer):
    """Fourier transformer.

    Parameters
    ----------
    sp : int
        Seasonal periodicity.
    n_terms : int, default=None
        The number of sine and cosine terms to include. For example,
        if `n_terms` = 2, 4 new columns will be generated. `n_terms` must not
        exceed `m / 2`. If None, `n_terms = sp // 2`.

    Notes
    -----
    This implementation is based on the `FourierFeaturizer` in pmdarima [3]_.

    References
    ----------
    .. [1] https://github.com/robjhyndman/forecast/blob/master/R/season.R
    .. [2] https://robjhyndman.com/hyndsight/longseasonality/
    .. [3] https://alkaline-ml.com/pmdarima/index.html

    See Also
    --------
    pmdarima.preprocessing.FourierFeaturizer, DateTimeFeatures
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "univariate-only": True,
        "handles-missing-data": True,
        "X_inner_mtype": ["pd.Series", "pd.DataFrame"],
        # "X_inner_mtype": "pd.Series",
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "fit-in-transform": False,
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
    }

    def __init__(self, sp, n_terms=None):
        self.sp = sp
        self.n_terms = n_terms
        super().__init__()

    def _fit(self, X, y=None):
        """
        Fit transformer to X and y.

        Parameters
        ----------
        X : Series
        y : Series, default=None

        Returns
        -------
        self: a fitted instance of the estimator
        """
        # Check parameters.
        check_sp(self.sp, allow_none=False)
        if self.n_terms is None:
            self.n_terms_ = self.sp // 2
        else:
            self.n_terms_ = _check_n_terms(self.n_terms)

        # Instantiate transformer from pmdarima.
        self._transformer = _FourierFeaturizer(m=self.sp, k=self.n_terms_)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : Series
        y : Series, default=None

        Returns
        -------
        Series
            Transformed X.
        """
        # Apply transformer from pmdarima.
        _, Xt = self._transformer.fit_transform(X)
        Xt.index = X.index
        return pd.concat([X, Xt], axis=1)


def _check_n_terms(n_terms):
    """Check the number of terms.

    Parameters
    ----------
    n_terms : int
        The number of terms.

    Returns
    -------
    int
        Checked number of terms.

    Raises
    ------
    ValueError
        If n_terms is not a positive integer.
    """
    if not isinstance(n_terms, (int, np.integer)) and n_terms > 0:
        raise ValueError(f"n_terms must be a positive integer, but found: {n_terms}")

    return n_terms
