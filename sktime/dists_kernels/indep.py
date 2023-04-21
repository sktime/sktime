# -*- coding: utf-8 -*-
"""Wrappers to convert distance to kernel or kernel to distance."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels._base import BasePairwiseTransformerPanel

SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ"]


class IndepDist(BasePairwiseTransformerPanel):
    r"""Kernel function obtained from a distance function.

    Formal details (for real valued objects, mixed typed rows in analogy):
    Let :math:`d: \mathbb{R}^D \times \mathbb{R}^D\rightarrow \mathbb{R}`
    be the pairwise function in `dist`, when applied to `D`-vectors.
    If `dist_diag=None`, then `KernelFromDist(dist)` corresponds to the kernel function
    :math:`k(x, y) := d(x, x)^2 + d(y, y)^2 - 0.5 \cdot d(x, y)^2`.
    If `dist_diag` is provided,
    and corresponds to a function :math:`f:\mathbb{R}^D \rightarrow \mathbb{R}`,
    then `KernelFromDist(dist)` corresponds to the kernel function
    :math:`k(x, y) := f(x, x)^2 + f(y, y)^2 - 0.5 \cdot d(x, y)^2`.

    It should be noted that :math:`k` is, in general, not positive semi-definite.

    Parameters
    ----------
    dist : pairwise transformer of BasePairwiseTransformer scitype, or
        callable np.ndarray (n_samples, nd) x (n_samples, nd) -> (n_samples x n_samples)
    """

    _tags = {
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
    }

    def __init__(self, dist):

        self.dist = dist

        super(IndepDist, self).__init__()

        # set property tags based on tags of components
        missing = True
        multi = True
        unequal = True
        if isinstance(dist, BasePairwiseTransformerPanel):
            missing = missing and dist.get_tag("capability:missing_values")
            multi = multi and dist.get_tag("capability:multivariate")
            unequal = unequal and dist.get_tag("capability:unequal_length")

        tag_dict = {
            "capability:missing_values": missing,
            "capability:multivariate": multi,
            "capability:unequal_length": unequal,
        }

        self.set_tags(**tag_dict)

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        private _transform containing core logic, called from public transform

        Parameters
        ----------
        X: sktime Panel data container
        X2: sktime Panel data container

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        from sktime.transformations.base import BaseTransformer

        dist = self.dist

        distmat = dist(X, X2) ** 2

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from sktime.dists_kernels.dtw import DtwDist
        from sktime.transformations.series.adapt import PandasTransformAdaptor
        from sktime.transformations.series.summarize import SummaryTransformer

        params1 = {"dist": DtwDist()}
        t = SummaryTransformer("mean", None)
        # we need this since multivariate summary produces two columns
        # if one column, has no effect; if multiple, takes means by row
        t = PandasTransformAdaptor("mean", {"axis": 1}) * t
        params2 = {"dist": DtwDist(), "dist_diag": t}

        return [params1, params2]
