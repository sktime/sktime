"""Delegator mixin that delegates all methods to wrapped transformer.

Useful for building estimators where all but one or a few methods are delegated.
For that purpose, inherit from this estimator and then override only the methods that
are not delegated.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["_DelegatedPairwiseTransformerPanel"]

from sktime.dists_kernels.base import BasePairwiseTransformerPanel


class _DelegatedPairwiseTransformerPanel(BasePairwiseTransformerPanel):
    """Delegator mixin that delegates all methods to wrapped transformer.

    Delegates inner transformer methods to a wrapped estimator.
        Wrapped estimator is value of attribute with name self._delegate_name.
        By default, this is "estimator_", i.e., delegates to self.estimator_
        To override delegation, override _delegate_name attribute in child class.

    Delegates the following inner underscore methods:
        _transform

    Does NOT delegate get_params, set_params.
        get_params, set_params will hence use one additional nesting level by default.

    Does NOT delegate or copy tags, this should be done in a child class if required.
    """

    # attribute for _DelegatedBasePairwiseTransformerPanel, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedBasePairwiseTransformerPanel docstring
    _delegate_name = "estimator_"

    def _get_delegate(self):
        return getattr(self, self._delegate_name)

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        private _transform containing core logic, called from transform

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2 (equal to X if not passed)

        Parameters
        ----------
        X : guaranteed to be Series or Panel of mtype X_inner_mtype, n instances
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        X2 : guaranteed to be Series or Panel of mtype X_inner_mtype, m instances
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]
        """
        estimator = self._get_delegate()
        return estimator.transform(X, X2=X2)
