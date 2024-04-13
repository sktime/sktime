#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements change point detection from ruptures library."""

from sktime.annotation.base._base import BaseSeriesAnnotator
from sktime.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["duydl"]


class RupturesKernelCPD(BaseSeriesAnnotator):
    r"""Creates a RupturesKernelCPD instance.

    Available kernels:

    - `linear`: $k(x,y) = x^T y$.
    - `rbf`: $k(x, y) = exp(\gamma \|x-y\|^2)$ where $\gamma>0$
    (`gamma`) is a user-defined parameter.
    - `cosine`: $k(x,y)= (x^T y)/(\|x\|\|y\|)$.

    Parameters
    ----------
        kernel (str, optional): name of the kernel, ["linear", "rbf", "cosine"]
        min_size (int, optional): minimum segment length.
        jump (int, optional): not considered, set to 1.
        params (dict, optional): a dictionary of parameters for the kernel instance
        n_bkps (int, optional): Number of change points. Defaults to None.
        pen (float, optional): penalty value (>0). Defaults to None. Not considered
            if n_bkps is not None.
    """

    _tags = {
        "univariate-only": True,
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
        _check_soft_dependencies("ruptures", severity="error")

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
        X : arraylike (1D np.ndarray or pd.series), shape = [num_observations]
            Observations to apply labels.

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
        X : 1D np.array, shape = [num_observations]
            Observations to apply labels to.

        Returns
        -------
        cps :
            A sorted list of change points.
        """
        cps = self._estimator.fit_predict(X, n_bkps=self._n_cps, pen=self._pen)
        return cps

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


# class RupturesBinsegCPD(BaseSeriesAnnotator):
#     pass


# class RupturesWindowCPD(BaseSeriesAnnotator):
#     pass


# class RupturesBottomUpCPD(BaseSeriesAnnotator):
#     pass
