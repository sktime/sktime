"""Identity transformer.

Note to developers: this is used as a component in many other transformers, therefore
one should avoid importing other transformers from here.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["Id"]

from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose._common import CORE_MTYPES


class Id(BaseTransformer):
    """Identity transformer, returns data unchanged in transform/inverse_transform."""

    _tags = {
        "authors": "fkiraly",
        "capability:inverse_transform": True,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": CORE_MTYPES,  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
        "handles-missing-data": True,  # can estimator handle missing data?
    }

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : any sktime compatible data, Series, Panel, or Hierarchical
        y : optional, default=None
            ignored, argument present for interface conformance

        Returns
        -------
        X, identical to input
        """
        return X

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : any sktime compatible data, Series, Panel, or Hierarchical
        y : optional, default=None
            ignored, argument present for interface conformance

        Returns
        -------
        X, identical to input
        """
        return X

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        return {}
