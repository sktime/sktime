"""Arithmetic with distances/kernels, e.g., addition, multiplication."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.base import _HeterogenousMetaEstimator
from sktime.dists_kernels.base import BasePairwiseTransformerPanel

SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ", "df-list", "numpy3D"]


class CombinedDistance(_HeterogenousMetaEstimator, BasePairwiseTransformerPanel):
    """Distances combined via arithmetic operation, e.g., addition, multiplication.

    ``CombinedDistance`` creates a pairwise trafo from multiple other pairwise trafos,
    by performing an arithmetic operation (np.ufunc) on the multiple distance matrices.

    For a list of transformers ``trafo1``, ``trafo2``, ..., ``trafoN``, ufunc
    ``operation``,
    this compositor behaves as follows:
    ``transform(X, X2)`` - computes ``dist1 = trafo1.transform(X, X2)``,
        ``dist2 = trafo2.transform(X, X2)``, ..., distN =  trafoN.transform(X, X2)`,
        all of shape ``(len(X), len(X2)``, then applies ``operation`` entry-wise,
        to obtain a single matrix ``dist`` of shape ``(len(X), len(X2)``
        Example: if ``operation = np.sum``, then ``dist`` is
        the entry-wise sum of ``dist1``, ``dist2``, ..., ``distN``

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

    the same can also be done more compactly using dunders:

    >>> sum_dist = DtwDist() + DtwDist(weighted=True)
    >>> dist_mat = sum_dist(X)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        # estimator type
        # --------------
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_pw_trafos"

    def __init__(self, pw_trafos, operation=None):
        self.pw_trafos = pw_trafos
        self.pw_trafos_ = self._check_estimators(
            self.pw_trafos, cls_type=BasePairwiseTransformerPanel
        )
        self.operation = operation
        self._operation = self._resolve_operation(operation)

        super().__init__()

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

    def _algebra_dunder_concat(self, other, operation):
        """Return (right) concat CombinedDistance, common boilerplate for dunders.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` pairwise transformer, must inherit
        BasePairwiseTransformerPanel
            otherwise, ``NotImplemented`` is returned
        operation: operation string used in CombinedDistance for the dunder.
            Must be equal to the operation of the dunder, not of self.

        Returns
        -------
        CombinedDistance object, concat of ``self`` (first) with ``other`` (last).
            does not contain CombinedDistance ``sktime`` transformers with same
            operation
            (but may nest CombinedDistance with different operations)
        """
        if self.operation == operation:
            # if other is CombinedDistance but with different operation,
            # we need to wrap it, or _dunder_concat would overwrite the operation
            if isinstance(other, CombinedDistance) and not other.operation == operation:
                other = CombinedDistance([other], operation=operation)
            return self._dunder_concat(
                other=other,
                base_class=BasePairwiseTransformerPanel,
                composite_class=CombinedDistance,
                attr_name="pw_trafos",
                concat_order="left",
                composite_params={"operation": operation},
            )
        elif isinstance(other, BasePairwiseTransformerPanel):
            return CombinedDistance([self, other], operation=operation)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Magic * method, return (right) multiplied CombinedDistance.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` pairwise transformer, must inherit
        BasePairwiseTransformerPanel
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        CombinedDistance object, algebraic * of ``self`` (first) with ``other`` (last).
            does not contain CombinedDistance ``sktime`` transformers with same
            operation
            (but may nest CombinedDistance with different operations)
        """
        return self._algebra_dunder_concat(other=other, operation="*")

    def __add__(self, other):
        """Magic + method, return (right) multiplied CombinedDistance.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` pairwise transformer, must inherit
        BasePairwiseTransformerPanel
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        CombinedDistance object, algebraic + of ``self`` (first) with ``other`` (last).
            not nested, contains only non-CombinedDistance ``sktime`` transformers
        """
        return self._algebra_dunder_concat(other=other, operation="+")

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sktime.dists_kernels.compose_tab_to_panel import AggrDist
        from sktime.dists_kernels.dtw import DtwDist

        params1 = {"pw_trafos": [AggrDist.create_test_instance()]}
        params2 = {
            "pw_trafos": [AggrDist.create_test_instance(), DtwDist()],
            "operation": "+",
        }

        return [params1, params2]
