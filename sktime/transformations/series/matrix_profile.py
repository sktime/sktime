#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements matrix profile transformation."""

__author__ = ["mloning"]
__all__ = ["MatrixProfileTransformer"]

from sktime.transformations.base import BaseTransformer


class MatrixProfileTransformer(BaseTransformer):
    """Calculate the matrix profile of a time series.

    Takes as input a single time series dataset and returns the matrix profile
    for that time series dataset. The matrix profile is a vector that stores the
    z-normalized Euclidean distance between any subsequence within a
    time series and its nearest neighbor.

    For more information on the matrix profile, see `stumpy's tutorial
    <https://stumpy.readthedocs.io/en/latest/Tutorial_The_Matrix_Profile.html>`_

    Parameters
    ----------
    window_length : int

    Notes
    -----
    Provides wrapper around functionality in `stumpy.stump
    <https://stumpy.readthedocs.io/en/latest/api.html#stumpy.stump>`_

    Examples
    --------
    >>> from sktime.transformations.series.matrix_profile import \
    MatrixProfileTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = MatrixProfileTransformer()  # doctest: +SKIP
    >>> y_hat = transformer.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["seanlaw", "NimaSarajpoor", "mloning"],
        "python_dependencies": ["stumpy", "numpy<2"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["np.ndarray"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?,
        "capability:multivariate": False,
        "fit_is_empty": True,  # for unit test cases
        # testing configuration
        # ---------------------
        "tests:vm": True,  # run in VM due to dependency requirement stumpy
    }

    def __init__(self, window_length=3):
        self.window_length = window_length
        super().__init__()

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
        Xt : 1D np.ndarray
            transformed version of X
            Matrix Profile of time series as output with length as
            (n_timepoints-window_length+1)
        """
        import numpy as np
        import stumpy

        X = X.flatten()
        Xt = stumpy.stump(X, self.window_length)
        Xt = np.asarray(Xt[:, 0].astype("float"))
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {}
        params1 = {"window_length": 5}
        return [params0, params1]
