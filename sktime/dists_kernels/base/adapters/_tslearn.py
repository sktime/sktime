# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for tslearn distances and kernels."""
import numpy as np

__all__ = ["_TslearnPwTrafoAdapter"]
__author__ = ["fkiraly"]


def _subset_dict(d, keys):
    """Subsets dictionary to keys in iterable keys.

    Parameters
    ----------
    d : dict
        dictionary to subset
    keys : iterable
        keys to subset to

    Returns
    -------
    dict
        subsetted dictionary
    """
    return {key: d[key] for key in keys if key in d}


class _TslearnPwTrafoAdapter:
    """Base adapter mixin for tslearn distances and kernels."""

    _tags = {
        "symmetric": False,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
        "X_inner_mtype": "df-list",
        # which mtype is used internally in _transform?
        "fit_is_empty": True,  # is "fit" empty? Yes, for all pairwise transforms
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "pwtrafo_type": "distance",  # type of pw. transformer, "kernel" or "distance"
        "python_dependencies": ["tslearn"],
    }

    # parameters to pass to the inner tslearn estimator, list of str
    # if None, will pass all of self.get_params()
    # otherwise, passes only the parameters in the list of str _inner_params
    _inner_params = None

    # controls whether vectorization is applied to the tslearn pwtrafo
    # True: the adapted function is cdist-like, it can take Panel data directly
    # False: the adapted function takes two time series and needs to be vectorized
    _is_cdist = True

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
        params = self.get_params()
        if self._inner_params is not None:
            params = _subset_dict(params, self._inner_params)

        return pwtrafo(X, X2, **params)

    def _coerce_df_list_to_list_of_arr(self, X):
        return [df.values for df in X]

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

        m = len(X)
        n = len(X2)
        res = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                res[i, j] = self._eval_tslearn_pwtrafo(X[i], X2[j])
        return res

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
        if isinstance(X, list):
            X = self._coerce_df_list_to_list_of_arr(X)
        if isinstance(X2, list):
            X2 = self._coerce_df_list_to_list_of_arr(X2)

        if self._is_cdist:
            return self._eval_tslearn_pwtrafo(X, X2)
        else:
            return self._eval_tslearn_pwtrafo_vectorized(X, X2)
