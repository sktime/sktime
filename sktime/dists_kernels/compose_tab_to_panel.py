# -*- coding: utf-8 -*-
"""
Composers that create panel pairwise transformers from table pairwise transformers.

Currently implemented composers in this module:

    AggrDist - panel distance from aggregation of tabular distance matrix entries
    FlatDist - panel distance from applying tabular distance to flattened panel matrix
"""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels._base import BasePairwiseTransformerPanel
from sktime.utils._testing.deep_equals import deep_equals


class AggrDist(BasePairwiseTransformerPanel):
    """Panel distance from tabular distance aggregation.

    panel distance obtained by applying aggregation function to tabular distance matrix
        example: AggrDist(ScipyDist()) is mean Euclidean distance between series

    Parameters
    ----------
    transformer: pairwise transformer of BasePairwiseTransformer scitype
    aggfunc: aggregation function (2D np.array) -> float or None, optional
        default = None = np.mean
    aggfunc_is_symm: bool, optional, default=False
        whether aggregation function is symmetric (should be set according to aggfunc)
            i.e., invariant under transposing argument, it always holds that
                aggfunc(matrix) = aggfunc(np.transpose(matrix))
            used for fast computation of the resultant matrix (if symmetric)
            if unknown, False is the "safe" option that ensures correctness
    """

    def __init__(
        self,
        transformer,
        aggfunc=None,
        aggfunc_is_symm=False,  # False for safety, but set True later if aggfunc=None
    ):

        self.aggfunc = aggfunc
        self.aggfunc_is_symm = aggfunc_is_symm
        self.transformer = transformer

        super(AggrDist, self).__init__()

        if self.aggfunc_is_symm:
            self.set_tag("symmetric", True)

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

            Core logic.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        n = len(X)
        m = len(X2)

        X_equals_X2 = deep_equals(X, X2)

        aggfunc = self.aggfunc
        aggfunc_is_symm = self.aggfunc_is_symm
        if aggfunc is None:
            aggfunc = np.mean
            aggfunc_is_symm = True

        transformer_symm = self.transformer.get_tag("symmetric", False)

        # whether we know that resulting matrix must be symmetric
        # a sufficient condition for this:
        # transformer is symmetric; X equals X2; and aggfunc is symmetric
        all_symm = aggfunc_is_symm and transformer_symm and X_equals_X2

        distmat = np.zeros((n, m), dtype="float")

        for i in range(n):
            for j in range(m):

                if all_symm and j < i:
                    distmat[i, j] = distmat[j, i]
                else:
                    distmat[i, j] = aggfunc(self.transformer.transform(X[i], X2[j]))

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for AggrDist."""
        # importing inside to avoid circular dependencies
        from sktime.dists_kernels import ScipyDist

        return {"transformer": ScipyDist()}


class FlatDist(BasePairwiseTransformerPanel):
    """Panel distance from applying tabular distance to flattened time series.

    Applies the wrapped tabular distance to flattened series.
    Flattening is done to a 2D numpy array of shape (n_instances, (n_vars, n_timepts))

    Parameters
    ----------
    transformer: pairwise transformer of BasePairwiseTransformer scitype
    """

    _tags = {
        "X_inner_mtype": "numpy3D",  # which mtype is used internally in _transform?
    }

    def __init__(self, transformer):

        self.transformer = transformer

        super(FlatDist, self).__init__()

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

            Core logic.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        n_inst, n_vars, n_tts = X.shape
        X = X.reshape(n_inst, n_vars * n_tts)

        n_inst2, n_vars2, n_tts2 = X2.shape
        X2 = X2.reshape(n_inst2, n_vars2 * n_tts2)

        if deep_equals(X, X2):
            return self.transformer.transform(X)
        else:
            return self.transformer.transform(X2)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for FlatDist."""
        # importing inside to avoid circular dependencies
        from sktime.dists_kernels import ScipyDist

        return {"transformer": ScipyDist()}
