"""
Composers that create panel pairwise transformers from table pairwise transformers
"""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels._base import BaseTrafoPwPanel


class AggrDist(BaseTrafoPwPanel):
    """
    pairwise distance based on aggregation function applied to sample-wise distance
        example: AggrDist(ScipyDist()) is mean Euclidean distance between series

    Components
    ----------
    transformer: pairwise transformer of BaseTrafoPw scitype (tabular pairwise)

    Hyper-parameters
    ----------------
    aggfun: aggregation function 2D np.array -> float
        default = np.mean
    """

    def __init__(self,
                 transformer,
                 aggfun=None,
                 ):

        if aggfun is None:
            aggfun = np.mean

        self.aggfun = aggfun

        self.transformer = transformer

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

        symmetric = self.symmetric
        aggfun = self.aggfun

        distmat = np.zeros((n, m), dtype="float")

        for i in range(n):
            for j in range(m):

                if symmetric and j < i:
                    distmat[i, j] = distmat[j, i]
                else:
                    distmat[i, j] = aggfun(self.transformer.transform(X[i], X2[j]))

        return distmat
