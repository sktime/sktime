# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Fourier features."""

__author__ = ["mloning"]

import numpy as np

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
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit-in-transform": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
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
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        check_sp(self.sp, allow_none=False)

        if self.n_terms is None:
            self.n_terms_ = self.sp // 2
        else:
            self.n_terms_ = _check_n_terms(self.n_terms)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : Series
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Series
            Transformed X.
        """
        pass


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
