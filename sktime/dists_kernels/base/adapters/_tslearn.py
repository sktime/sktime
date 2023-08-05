# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for tslearn distances and kernels."""
import numpy as np

__all__ = ["_TslearnPwTrafoAdapter"]
__author__ = ["fkiraly"]


class _TslearnPwTrafoAdapter:
    """Base adapter mixin for tslearn distances and kernels."""

    _tags = {
        "symmetric": False,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
        "X_inner_mtype": "numpy3D",  # which mtype is used internally in _transform?
        "fit_is_empty": True,  # is "fit" empty? Yes, for all pairwise transforms
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "pwtrafo_type": "distance",  # type of pw. transformer, "kernel" or "distance"
        "python_dependencies": ["tslearn"],
    }

    def _get_tslearn_pwtrafo(self):
        """Abstract method to get tslearn pwtrafo.

        should import and return tslearn pwtrafo
        """
        # from tslearn import pwtrafo
        #
        # return pwtrafo
        raise NotImplementedError("abstract method")

    def _eval_tslearn_pwtrafo(self, X, X2=None):
        """Evaluate tslearn pwtrafo on two time series.

        The default returns of _get_tslearn_pwtrafo
        evaluated at X1, X2 and self.get_params

        Parameters
        ----------
        X, X2: 2D np.ndarrays of format (n_variables, n_timepoints)
            two time series to compute the pairwise transform on

        Returns
        -------
        float
            _get_tslearn_pwtrafo result evaluated at X1, X2, and self.get_params()
        """
        if X2 is None:
            X2 = X

        pwtrafo = self._get_tslearn_pwtrafo()
        return pwtrafo(X, X2, **self.get_params())

    def _eval_tslearn_pwtrafo_vectorized(self, X, X2=None):
        """Evaluate tslearn pwtrafo on two time series panels.

        Vectorizes _eval_tslearn_pwtrafo over the first dimensions.

        Parameters
        ----------
        X, X2: 3D np.ndarrays of format (n_instances n_variables, n_timepoints)
            two time series panels to compute the pairwise transform on

        Returns
        -------
        2D np.ndarray
            (i, j)-th entry is _eval_tslearn_pwtrafo(self, X1[i], X2[j])
        """
        if X2 is None:
            X2 = X

        dist_kern_vectorized = np.vectorize(
            self._eval_tslearn_pwtrafo,
            signature="(m,k,l),(n,k,l)->(m,n)"
        )
        return dist_kern_vectorized(X, X2)

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

            Core logic

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: 3D np.array of shape [num_instances, num_vars, num_time_points]
        X2: 3D np.array of shape [num_instances, num_vars, num_time_points], optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]
        """
        return self._eval_tslearn_pwtrafo_vectorized(self, X, X2)
