#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements change point detection from ruptures library."""

from sktime.annotation.base._base import BaseSeriesAnnotator
from sktime.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["duydl"]
import numpy as np
import pandas as pd

_check_soft_dependencies("ruptures", severity="error")


class RupturesKernelCPD(BaseSeriesAnnotator):
    r"""Creates a RupturesKernelCPD instance.

    Available kernels:

    - `linear`: $k(x,y) = x^T y$.
    - `rbf`: $k(x, y) = \exp(\gamma \|x-y\|^2)$ where $\gamma>0$
      (`gamma`) is a user-defined parameter.
    - `cosine`: $k(x,y)= (x^T y)/(\|x\|\|y\|)$.

    Parameters
    ----------
    kernel : str, optional
        Name of the kernel, ["linear", "rbf", "cosine"].
    min_size : int, optional
        Minimum segment length.
    jump : int, optional
        Not considered, set to 1.
    params : dict, optional
        A dictionary of parameters for the kernel instance.
    n_cps : int, optional
        Number of change points. Defaults to None.
    pen : float, optional
        Penalty value (>0). Defaults to None. Not considered
        if n_cps is not None.

    Attributes
    ----------
    _tags : dict
        Tags for the estimator indicating its properties.
    _estimator : ruptures.KernelCPD
        Instance of the Ruptures KernelCPD estimator.
    _n_cps : int, optional
        Number of change points.
    _pen : float, optional
        Penalty value.

    See Also
    --------
    ruptures.KernelCPD : Ruptures library's KernelCPD estimator.

    Examples
    --------
    >>> from sktime.annotation.adapters._ruptures import RupturesKernelCPD
    >>> from sktime.datasets import load_gun_point_segmentation
    >>> X, true_period_size, cps = load_gun_point_segmentation() # doctest: +SKIP
    >>> kernelcpd = RupturesKernelCPD(dominant_period_size, n_cps=1) # doctest: +SKIP
    >>> found_cps = kernelcpd.fit_predict(X) # doctest: +SKIP
    """

    _tags = {
        "univariate-only": False,
        "fit_is_empty": True,
        "python_dependencies": "ruptures",
    }

    def __init__(
        self,
        kernel="linear",
        min_size=2,
        jump=1,
        n_cps=None,
        pen=None,
        params=None,
    ):
        import ruptures

        self._estimator = ruptures.KernelCPD(
            kernel=kernel, min_size=min_size, jump=jump, params=params
        )
        self._n_cps = n_cps
        self._pen = pen
        super().__init__()

    def _fit(self, X, Y=None):
        """Fit the wrapped estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features) or (n_samples,)
            Input data. Multivariate time series data where each row represents
            a sample, and each column represents a feature. If 1D, it represents
            univariate time series.

        Returns
        -------
        self :
            Returns a reference to self.
        """
        return self

    def _predict(self, X):
        """Ensure the input type is correct, then predict using wrapped estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features) or (n_samples,)
            Input data. Multivariate time series data where each row represents
            a sample, and each column represents a feature. If 1D, it represents
            univariate time series.

        Returns
        -------
        cps : array-like
            A sorted list of change points.
        """
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        cps = self._estimator.fit_predict(X, n_bkps=self._n_cps, pen=self._pen)
        return np.array(cps[:-1])

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

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
        return [{"pen": None, "n_cps": 1}, {"pen": 5, "n_cps": None}]


class RupturesBinsegCPD(BaseSeriesAnnotator):
    """Adapter for Ruptures Binseg.

    Parameters
    ----------
    model (str, optional): segment model, ["l1", "l2", "rbf",...].
        Not used if ``'custom_cost'`` is not None.
    custom_cost (BaseCost, optional): custom cost function. Defaults to None.
    min_size (int, optional): minimum segment length. Defaults to 2 samples.
    jump (int, optional): subsample (one every *jump* points).
    n_cps (int): Number of breakpoints to find before stopping.
    pen (float): Penalty value (>0).
    epsilon (float): Reconstruction budget (>0).
    params (dict, optional): a dictionary of parameters for the cost instance.
    """

    _tags = {
        "univariate-only": False,
        "fit_is_empty": True,
        "python_dependencies": "ruptures",
    }

    def __init__(
        self,
        model="l2",
        custom_cost=None,
        min_size=2,
        jump=1,
        n_cps=None,
        pen=None,
        epsilon=None,
        params=None,
    ):
        import ruptures

        self._estimator = ruptures.Binseg(
            model=model,
            custom_cost=custom_cost,
            min_size=min_size,
            jump=jump,
            params=params,
        )
        self._n_cps = n_cps
        self._pen = pen
        self._epsilon = epsilon
        super().__init__()

    def _fit(self, X, Y=None):
        """Fit the Binseg instance to the input data.

        Parameters
        ----------
            X (array-like): Input data. Shape (n_samples, n_features) or (n_samples,).

        Returns
        -------
            self
        """
        return self

    def _predict(self, X):
        """Predict the optimal breakpoints using the fitted Binseg instance.

        Parameters
        ----------
            X (array-like): Input data. Shape (n_samples, n_features) or (n_samples,).

        Returns
        -------
            list: Sorted list of breakpoints.
        """
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        cps = self._estimator.fit_predict(
            X, n_bkps=self._n_cps, pen=self._pen, epsilon=self._epsilon
        )
        return np.array(cps[:-1])


class RupturesWindowCPD(BaseSeriesAnnotator):
    """Adapter for Ruptures Window."""

    def __init__(
        self,
        width=100,
        model="l2",
        custom_cost=None,
        min_size=2,
        jump=1,
        n_cps=None,
        pen=None,
        params=None,
    ):
        import ruptures

        super().__init__()

        self._estimator = ruptures.Window(
            width=width,
            model=model,
            custom_cost=custom_cost,
            min_size=min_size,
            jump=jump,
            params=params,
        )
        self._width = width
        self._n_cps = n_cps
        self._pen = pen

    def _fit(self, X, Y=None):
        """Fit the Window instance to the input data.

        Parameters
        ----------
            X (array-like): Input data. Shape (n_samples, n_features) or (n_samples,).

        Returns
        -------
            self
        """
        return self

    def _predict(self, X):
        """Predict the optimal breakpoints using the fitted Window instance.

        Parameters
        ----------
            X (array-like): Input data. Shape (n_samples, n_features) or (n_samples,).

        Returns
        -------
            list: Sorted list of breakpoints.
        """
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        cps = self._estimator.fit_predict(X, n_bkps=self._n_cps, pen=self._pen)
        return np.array(cps[:-1])


class RupturesBottomUpCPD(BaseSeriesAnnotator):
    """Adapter for Ruptures BottomUp."""

    def __init__(
        self,
        model="l2",
        custom_cost=None,
        min_size=2,
        jump=1,
        n_cps=None,
        pen=None,
        params=None,
    ):
        import ruptures

        super().__init__()

        self._estimator = ruptures.BottomUp(
            model=model,
            custom_cost=custom_cost,
            min_size=min_size,
            jump=jump,
            params=params,
        )

        self._n_cps = n_cps
        self._pen = pen

    def _fit(self, X, Y=None):
        """Fit the Window instance to the input data.

        Parameters
        ----------
            X (array-like): Input data. Shape (n_samples, n_features) or (n_samples,).

        Returns
        -------
            self
        """
        return self

    def _predict(self, X):
        """Predict the optimal breakpoints using the fitted Window instance.

        Parameters
        ----------
            X (array-like): Input data. Shape (n_samples, n_features) or (n_samples,).

        Returns
        -------
            list: Sorted list of breakpoints.
        """
        # if isinstance(X, pd.Series):
        #     X = X.to_numpy()
        cps = self._estimator.fit_predict(X, n_bkps=self._n_cps, pen=self._pen)
        return np.array(cps[:-1])
