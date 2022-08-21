# -*- coding: utf-8 -*-
"""Arithmetics with distances/kernels, e.g., addition, multiplication."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.base import _HeterogenousMetaEstimator
from sktime.dists_kernels._base import BasePairwiseTransformerPanel

SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ", "df-list", "numpy3D"]


class CombinedDistance(BasePairwiseTransformerPanel, _HeterogenousMetaEstimator):
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
    >>> from sktime.dists_kernels.dtw import DtwDist
    >>> from sktime.datasets import load_unit_test
    >>>
    >>> X, _ = load_unit_test()
    >>> X = X[0:3]
    >>> sum_dist = CombinedDistance([DtwDist(), DtwDist(weighted=True)], "+")
    >>> dist_mat = sum_dist.transform(X)
    """

    _tags = {
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
    }

    def __init__(self, pw_trafos, operation=None):

        self.pw_trafos = pw_trafos
        self.pw_trafos_ = self._check_estimators(
            self.pw_trafos, cls_type=BasePairwiseTransformerPanel
        )
        self.operation = operation
        self._operation = self._resolve_operation(operation)

        super(CombinedDistance, self).__init__()

        # abbreviate for readability
        ests = self.pw_trafos_

        # set property tags based on tags of components
        self._anytagis_then_set("fit_is_empty", False, True, ests)
        self._anytagis_then_set("capability:missing_values", False, True, ests)
        self._anytagis_then_set("capability:multivariate", False, True, ests)
        self._anytagis_then_set("capability:unequal_length", False, True, ests)

    def _resolve_operation(self, operation):
        """Coerce operation to a numpy.ufunc."""
        alias_dict = {
            None: np.mean,
            "mean": np.mean,
            "+": np.add,
            "add": np.add,
            "*": np.multiply,
            "mult": np.multiply,
            "multiply": np.multiply,
            "min": np.min,
            "max": np.max,
        }

        if operation in alias_dict.keys():
            return alias_dict[operation]
        else:
            return operation

    @property
    def _pw_trafos(self):
        return self._get_estimator_tuples(self.pw_trafos, clone_ests=False)

    @_pw_trafos.setter
    def _pw_trafos(self, value):
        self.pw_trafos = value

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
        ests = self._get_estimator_list(self.pw_trafos_)

        distmats = [est.transform(X=X, X2=X2) for est in ests]
        distmat_stack = np.stack(distmats)

        operation = self._operation
        if isinstance(operation, np.ufunc):
            distmat = operation.reduce(distmat_stack)
        else:
            distmat = np.apply_over_axes(operation, distmat_stack, 0)
            distmat = distmat.squeeze(axis=0)

        return distmat

    def get_params(self, deep=True):
        """Get parameters of estimator in `steps`.

        Parameters
        ----------
        deep : boolean, optional, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("_pw_trafos", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `steps`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params("_pw_trafos", **kwargs)
        return self

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
        from sktime.dists_kernels.compose_tab_to_panel import AggrDist
        from sktime.dists_kernels.dtw import DtwDist

        params1 = {"pw_trafos": [AggrDist.create_test_instance()]}
        params2 = {
            "pw_trafos": [AggrDist.create_test_instance(), DtwDist()],
            "operation": "+",
        }

        return [params1, params2]
