# -*- coding: utf-8 -*-
"""
Composers that create panel pairwise transformers from table pairwise transformers
"""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels._base import BasePairwiseTransformerPanel


class AggrDist(BasePairwiseTransformerPanel):
    """
    panel distance obtained by applying aggregation function to tabular distance matrix
        example: AggrDist(ScipyDist()) is mean Euclidean distance between series

    Components
    ----------
    transformer: pairwise transformer of BasePairwiseTransformer scitype

    Hyper-parameters
    ----------------
    aggfunc: aggregation function 2D np.array -> float
        default = None, however, if transform is called then defaults to np.mean
    aggfunc_is_symm: bool - whether aggregation function is symmetric
                i.e., invariant under transposing argument, it always holds that
                    aggfunc(matrix) = aggfunc(np.transpose(matrix))
            used for fast computation of the resultant matrix (if symmetric)
            if unknown, False is the "safe" option that ensures correctness
        default = True if aggfunc default is used
            False otherwise (should be set according to choice of aggfunc)
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

    def _transform(self, X, X2=None):
        """
        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Args:
            X: list of pd.DataFrame or 2D np.arrays, of length n

        Optional args:
            X2: list of pd.DataFrame or 2D np.arrays, of length m

        Returns:
            distmat: np.array of shape [n, m]
                (i,j)-th entry contains distance/kernel between X[i] and X2[j]
        """

        n = len(X)
        m = len(X2)

        X_equals_X2 = self.X_equals_X2

        aggfunc = self.aggfunc
        aggfunc_is_symm = self.aggfunc_is_symm
        if aggfunc is None:
            aggfunc = np.mean
            aggfunc_is_symm = True

        transformer_symm = self.transformer._all_tags()["symmetric"]

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
