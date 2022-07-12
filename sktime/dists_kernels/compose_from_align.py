# -*- coding: utf-8 -*-
"""Composer that creates distance from aligner."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels._base import BasePairwiseTransformerPanel


class DistFromAligner(BasePairwiseTransformerPanel):
    """Distance transformer from aligner.

    Behaviour: uses aligner.get_distance on pairs to obtain distance matrix.

    Components
    ----------
    aligner: BaseAligner, must implement get_distances method
        if None, distance is equal zero
    """

    _tags = {
        "symmetric": True,  # all the distances are symmetric
    }

    def __init__(self, aligner=None):

        self.aligner = aligner

        super(DistFromAligner, self).__init__()

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
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        # find out whether we know that the resulting matrix is symmetric
        #   since aligner distances are always symmetric,
        #   we know it's the case for sure if X equals X2
        if X2 is None:
            X = X2
            symm = True
        else:
            symm = False

        n = len(X)
        m = len(X2)

        distmat = np.zeros((n, m), dtype="float")

        if self.aligner is not None:
            aligner = self.aligner.clone()
        else:
            return distmat

        for i in range(n):
            for j in range(m):
                if symm and j < i:
                    distmat[i, j] = distmat[j, i]
                else:
                    distmat[i, j] = aligner.fit([X[i], X2[j]]).get_distance()

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for DistFromAligner."""
        # importing inside to avoid circular dependencies
        from sktime.alignment.dtw_python import AlignerDTW
        from sktime.utils.validation._dependencies import _check_estimator_deps

        if _check_estimator_deps(AlignerDTW, severity="none"):
            return {"aligner": AlignerDTW()}
        else:
            return {}
