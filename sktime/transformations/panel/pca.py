# -*- coding: utf-8 -*-
__author__ = ["Patrick Rockenschaub"]
__all__ = ["PCATransformer"]

import pandas as pd
from sklearn.decomposition import PCA

from sktime.transformations.base import _PanelToPanelTransformer
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.utils.validation.panel import check_X


class PCATransformer(_PanelToPanelTransformer):
    """Principal Components Analysis applied to panel of time seires.

    Provides a simple wrapper around ``sklearn.decomposition.PCA``.

    Parameters
    ----------
    n_components : int, float, str or None (default None)
        Number of principle components to retain. By default, all components
        are retained. See
        ``sklearn.decomposition.PCA`` documentation for a detailed
        description of all options.
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
        "univariate-only": True,
        "fit-in-transform": False,
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
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        N, _, n = X.shape
        X = X.reshape(N, n)

        # Transform the time series column into tabular format and
        # apply PCA to the tabular format
        self.pca.fit(X)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        N, _, n = X.shape
        X = X.reshape(N, n)

        # Transform X using the fitted PCA
        Xt = self.pca.transform(X)
        N, n = Xt.shape
        Xt = Xt.reshape(N, 1, n)

        return Xt
