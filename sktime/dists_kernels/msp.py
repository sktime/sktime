# -*- coding: utf-8 -*-
"""BaseEstimator interface to sktime dtw distances in distances module."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels._base import BasePairwiseTransformerPanel


SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ", "df-list", "numpy3D"]


class MspDist(BasePairwiseTransformerPanel):
    r"""Minimum spanning tree distance.

    Takes an arbitrary distance and computes a minimum spanning tree distance
    based on it.

    For provided distance `pw_trafo`, does the following in `transform`:

    1. compute the distance matrix `D` on the sample `X`
    2. compute the symmetric, adjacency matrix `A` of the minimum
       spanning tree of `D`.
    3. Compute `D-prime` from `A`, which has identical entries to `A` for 1-s,
       and entries for 0-s according to the `transitive` parameter.
    4. Return `D-prime`.

    Uses `scipy.sparse.csgraph.minimum_spanning_tree` to compute the minimum
    spanning tree.

    Parameters
    ----------
    pw_trafo : pairwise panel transformer,
        i.e., estimator inheriting from `BasePairwiseTransformerPanel`
        distance that the spanning tree distance is based on.
    transitive_entries : str, one of "sum" (default), "zero", "inf", "max"
        treatment of the off-diagonal elements not on the spanning tree
        "sum" - sum distance based on the spanning tree
        "zero" - entries not on the spanning tree contain zero
        "inf" - entries not on the spanning tree contain np.inf
        "max" - entries not on the spanning tree contain maximal spanning distance (sum)

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.dists_kernels.dtw import DtwDist
    >>> from sktime.dists_kernels.msp import MspDist
    >>>
    >>> X, _ = load_unit_test(return_type="pd-multiindex")  # doctest: +SKIP
    >>> d = DtwDist(weighted=True, derivative=True)  # doctest: +SKIP
    >>> mspd = MspDist(d)  # doctest: +SKIP
    >>> distmat = mspd.transform(X)  # doctest: +SKIP
    """

    _tags = {
        "symmetric": True,  # all the distances are symmetric
        "X_inner_mtype": SUPPORTED_MTYPES,
        "python_dependencies": "scipy",
    }

    def __init__(self, pw_trafo, transitive_entries="sum"):

        self.pw_trafo = pw_trafo
        self.transitive_entries = transitive_entries

        super(MspDist, self).__init__()

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
        from scipy.sparse.csgraph import minimum_spanning_tree

        pw_trafo = self.pw_trafo
        transitive_entries = self.transitive_entries

        distmat = pw_trafo.transform(X, X2)
        # add 1 to avoid ambiguity about zero. we substract later again
        # adding a constant does not change the spanning tree edges
        distmat = distmat + 1

        mspmat = minimum_spanning_tree(distmat)
        mspadj = mspmat > 0
        mspadj = mspadj.astype("int")

        if transitive_entries == "zero":
            nz = mspmat > 0
            resmat = mspmat - nz
            resmat = resmat.toarray()

        elif transitive_entries == "sum":
            resmat = _sum_spanning_tree(mspmat, mspadj)

        elif transitive_entries in ["inf", "max"]:
            nonmspmask = 1 - mspadj
            if transitive_entries == "inf":
                const = np.inf
            else:
                const = np.max(distmat) - 1

            resmat = distmat.toarray() - 1 + nonmspmask * const

        return resmat


def _sum_spanning_tree(mspmat, mspadj):
    # fixed matrices
    # --------------
    # mspadjs = symmetric adjacency matrix of msp
    mspadjs = mspadj.transpose() + mspadj
    # exppat = symmetric exponential distance matrix of msp
    exppat = mspmat.expm1() + mspmat.transpose().expm1()
    exppat = exppat + mspadjs

    # updated matrices
    # ----------------
    # current length of span paths
    k = 1
    # cur_nb = 1 if vertices are k edges away, for k-th loop
    mspadjsp = mspadjs
    msppowk = mspadjsp
    cur_nb = mspadjs
    # paths = at entries in cur_nb, logarithmic sum distance
    paths = exppat
    # res = at all entries up to k edges away, spanning tree distance
    res = mspmat.toarray() + mspmat.transpose().toarray()

    # loop, starts with k=2
    while np.any(cur_nb.toarray() > 0):
        k = k + 1
        msppowk = msppowk * mspadjsp
        mask = (msppowk == 1).astype("int")
        cur_nb = msppowk.multiply(mask)
        cur_nb.setdiag(0)
        pathleft = paths * exppat
        pathright = exppat * paths
        paths = (pathleft + pathright) / 2
        paths = paths.multiply(cur_nb)
        expincrm1 = paths - cur_nb
        incr = np.log1p(expincrm1)
        res = res + incr.toarray()

    return res
