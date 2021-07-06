# -*- coding: utf-8 -*-
"""
Interface module to scipy.spatial's pairwise distance function cdist
    exposes parameters as scikit-learn hyper-parameters
"""

__author__ = ["fkiraly"]

import pandas as pd

from scipy.spatial.distance import cdist

from sktime.dists_kernels._base import BasePairwiseTransformer


class ScipyDist(BasePairwiseTransformer):
    """
    computes pairwise distances using scipy.spatial.distance.cdist
        includes Euclidean distance and p-norm (Minkowski) distance
            note: weighted distances are not supported

    Hyper-parameters:
        metric: string or function, as in cdist; default = 'euclidean'
            if string, one of: 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
                'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
                'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                'sokalsneath', 'sqeuclidean', 'yule'
            if function, should have signature 1D-np.array x 1D-np.array -> float
        p: if metric='minkowski', the "p" in "p-norm", otherwise irrelevant
        colalign: string, one of 'intersect' (default), 'force-align', 'none'
            controls column alignment if X, X2 passed in fit are pd.DataFrame
            columns between X and X2 are aligned via column names
            if 'intersect', distance is computed on columns occurring both in X and X2,
                other columns are discarded; column ordering in X2 is copied from X
            if 'force-align', raises an error if the set of columns in X, X2 differs;
                column ordering in X2 is copied from X
            if 'none', X and X2 are passed through unmodified (no columns are aligned)
                note: this will potentially align "non-matching" columns
    """

    _tags = {
        "symmetric": True,  # all the distances are symmetric
    }

    def __init__(self, metric="euclidean", p=2, colalign="intersect"):

        self.metric = metric
        self.p = p
        self.colalign = colalign

        super(ScipyDist, self).__init__()

    def _transform(self, X, X2=None):
        """
        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Args:
            X: pd.DataFrame of length n, or 2D np.array of 'float' with n rows

        Optional args:
            X2: pd.DataFrame of length m, or 2D np.array of 'float' with m rows

        Returns:
            distmat: np.array of shape [n, m]
                (i,j)-th entry contains distance between X.iloc[i] and X2.iloc[j]
                    (non-numeric columns are removed before for DataFrame X/X2)
        """

        p = self.p
        metric = self.metric

        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes("number").to_numpy(dtype="float")

        if isinstance(X2, pd.DataFrame):
            X2 = X2.select_dtypes("number").to_numpy(dtype="float")

        if metric == "minkowski":
            distmat = cdist(XA=X, XB=X2, metric=metric, p=p)
        else:
            distmat = cdist(XA=X, XB=X2, metric=metric)

        return distmat
