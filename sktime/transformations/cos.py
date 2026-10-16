#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements cosine transformation."""

from math import pi

import numpy as np

from sktime.transformations.base import BaseTransformer

__author__ = ["afzal442"]
__all__ = ["CosineTransformer"]


class CosineTransformer(BaseTransformer):
    """Cosine transformation.

    This is a wrapper around numpy's cosine function (see :func:`numpy.cos`).

    The inverse transform is :func:`numpy.arccos`, which returns values in
    ``[0, pi]``. The transform is therefore invertible on ``[0, pi]`` only:
    outside it ``inverse_transform`` returns the value in ``[0, pi]`` with the
    same cosine, so ``-1.0`` comes back as ``1.0``.

    See Also
    --------
    numpy.cos

    Examples
    --------
    >>> from sktime.transformations.cos import CosineTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = CosineTransformer()
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "afzal442",
        "maintainers": "afzal442",
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:multivariate": True,
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "capability:inverse_transform": True,
        # np.arccos returns values in [0, pi], so arccos(cos(x)) == x holds there
        # and nowhere else: on [-pi, 0) it returns -x, not x.
        "capability:inverse_transform:range": [0, pi],
    }

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            transformed version of X
        """
        Xt = np.cos(X)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
            inverse transformed version of X
        """
        Xt = np.arccos(X)
        return Xt
