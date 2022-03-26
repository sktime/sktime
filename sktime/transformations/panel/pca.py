# -*- coding: utf-8 -*-
"""sklearn PCA applied after flattening series."""
__author__ = ["prockenschaub", "fkiraly"]
__all__ = ["PCATransformer"]

import numpy as np
from sklearn.decomposition import PCA

from sktime.transformations.base import BaseTransformer


class PCATransformer(BaseTransformer):
    """Principal Components Analysis applied to panel of time seires.

    Provides a simple wrapper around ``sklearn.decomposition.PCA``.

    Applies PCA to a panel [num_instances, num_variables, num_time_points] as follows:
        1. flattens panel to [num_instances, num_time_points*num_variables]
        2. if fit: fits sklearn.pca to flattened panel
           in transform: projects on first n_components principal components,
                then re-formats back to [num_instances, num_variables, num_time_points]

    Parameters
    ----------
    n_components : int, float, str or None (default None)
        Number principal components in projection
        Default = min(num_instances, num_variables * num_time_points)
        See ``sklearn.decomposition.PCA`` documentation for further documentation.
    **kwargs
        Additional parameters passed on to ``sklearn.decomposition.PCA``.
        See ``sklearn.decomposition.PCA``
        documentation for a detailed description of all options.
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "univariate-only": False,
        "fit_is_empty": False,
    }

    def __init__(self, n_components=None, **kwargs):
        self.n_components = n_components
        self.pca = PCA(self.n_components, **kwargs)
        super(PCATransformer, self).__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Panel data in 3D np.ndarray format [n_instances, n_variables, n_timepoints]
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        N, num_var, num_time = X.shape
        X = X.reshape(N, num_time * num_var)

        # Transform the time series column into tabular format and
        # apply PCA to the tabular format
        self.pca.fit(X)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Panel data in 3D np.ndarray format [n_instances, n_variables, n_timepoints]
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : Panel data in 3D np.ndarray format [n_instances, n_variables, n_timepoints]
            transformed version of X
        """
        N, num_var, num_time = X.shape
        X = X.reshape(N, num_time * num_var)

        # Transform X using the fitted PCA
        Xt = np.matmul(self.pca.transform(X), self.pca.components_)
        Xt = Xt.reshape(N, num_var, num_time)

        return Xt
