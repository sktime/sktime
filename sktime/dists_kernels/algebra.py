# -*- coding: utf-8 -*-
"""Arithmetics with distances/kernels, e.g., addition, multiplication."""

__author__ = ["fkiraly"]

from deprecated.sphinx import deprecated

import sktime.dists_kernels.distances.algebra as new_class_loc

SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ", "df-list", "numpy3D"]


# TODO: remove file in v0.15.0
@deprecated(
    version="0.13.4",
    reason="CombinedDistance has moved and this import will be removed in 0.15.0. Import from sktime.dists_kernels.distances",  # noqa: E501
    category=FutureWarning,
)
class CombinedDistance(new_class_loc.CombinedDistance):
    """Distances combined via arithmetic operation, e.g., addition, multiplication.

    `CombinedDistance` creates a pairwise trafo from multiple other pairwise trafos,
    by performing an arithmetic operation (np.ufunc) on the multiple distance matrices.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN`, ufunc `operation`,
    this compositor behaves as follows:
    `transform(X, X2)` - computes `dist1 = trafo1.transform(X, X2)`,
        `dist2 = trafo2.transform(X, X2)`, ..., distN =  trafoN.transform(X, X2)`,
        all of shape `(len(X), len(X2)`, then applies `operation` entry-wise,
        to obtain a single matrix `dist` of shape `(len(X), len(X2)`
        Example: if `operation = np.sum`, then `dist` is
        the entry-wise sum of `dist1`, `dist2`, ..., `distN`

    Parameters
    ----------
    pw_trafos : list of sktime pairwise panel distances, or
        list of tuples (str, transformer) of sktime pairwise panel distances
        distances combined to a single distance using the operation
    operation : None, str, function, or numpy ufunc, optional, default = None = mean
        if str, must be one of "mean", "+" (add), "*" (multiply), "max", "min"
        if func, must be of signature (1D iterable) -> float
        operation carried out on the distance matrices distances

    Examples
    --------
    >>> from sktime.dists_kernels.algebra import CombinedDistance
    >>> from sktime.dists_kernels.distances.dtw import DtwDist
    >>> from sktime.datasets import load_unit_test
    >>>
    >>> X, _ = load_unit_test()
    >>> X = X[0:3]
    >>> sum_dist = CombinedDistance([DtwDist(), DtwDist(weighted=True)], "+")
    >>> dist_mat = sum_dist.transform(X)
    """

    def __init__(self, pw_trafos, operation=None):
        super(CombinedDistance, self).__init__(pw_trafos=pw_trafos, operation=operation)
