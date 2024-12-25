"""Sklearn PCA applied after flattening series."""

__author__ = ["prockenschaub", "fkiraly"]
__all__ = ["PCATransformer"]

import numpy as np
from sklearn.decomposition import PCA

from sktime.transformations.base import BaseTransformer


class PCATransformer(BaseTransformer):
    """Principal Components Analysis applied to panel of time series.

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

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, default=False
        When True (False by default) the ``components_`` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on ``X.shape`` and
            ``n_components``: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            ``scipy.linalg.svd`` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            ``scipy.sparse.linalg.svds``. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.

    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).

    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).

    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.
    """

    _tags = {
        "authors": ["prockenschaub", "fkiraly"],
        "maintainers": ["prockenschaub"],
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

    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        self.pca = PCA(
            n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )

        super().__init__()

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
