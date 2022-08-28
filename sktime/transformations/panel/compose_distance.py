# -*- coding: utf-8 -*-
"""Compositors that use pairwise transformers for an ordinary transformstion."""

__author__ = ["fkiraly"]
__all__ = ["DistanceFeatures"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class DistanceFeatures(BaseTransformer):
    """Use distances to training series as features.

    In `transform`, returns tabular features as follows:
    for `i`-th series in `X`, returns all distances to series seen in `fit`
    `j` th column of `i`-th row is distance between `i`-th series in `transform`,
    and `j`-the series in `fit`. Column index is instance index in `fit`.
    If `fit` series was `Hierarchical`, hierarchy index is preserved.

    Parameters
    ----------
    distance : pairwise panel transformer inheriting from BasePairwiseTransformerPanel,
        or callable, or None, optional. default = None = FlatDist(ScipyDist())
    distance_mtype : str, or list of str optional. default = None.
        mtype that distance expects for X and X2, if a callable
            only set this if distance is not BasePairwiseTransformerPanel descendant
    flatten_hierarchy : bool, optional, default=False.
        whether hierarchy in `transform` return is flattened (using `__` concat),
        in case of a hierarchical series index seen in `fit`.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
    >>> X_train, y_train = load_unit_test(return_X_y=True, split="train")
    >>> X_test, y_test = load_unit_test(return_X_y=True, split="test")
    >>> regressor = KNeighborsTimeSeriesRegressor()
    >>> regressor.fit(X_train, y_train)
    KNeighborsTimeSeriesRegressor(...)
    >>> y_pred = regressor.predict(X_test)
    """

    _tags = {
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": "None",
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:transform-labels": "None",
        "scitype:instancewise": False,
        "capability:inverse_transform": False,
        "univariate-only": False,
        "requires_y": False,
        "enforce_index_type": None,
        "fit_is_empty": False,
        "X-y-must-have-same-index": False,
        "transform-returns-same-time-index": False,
        "skip-inverse-transform": False,
        "capability:unequal_length": True,
        "capability:unequal_length:removes": False,
        "handles-missing-data": True,
        "capability:missing_values:removes": False,
    }

    def __init__(self, distance=None, distance_mtype=None, flatten_hierarchy=False):
        self.distance = distance
        self.distance_mtype = distance_mtype
        self.flatten_hierarchy = flatten_hierarchy

        super(DistanceFeatures, self).__init__()

        if distance_mtype is not None:
            self.set_tags(X_inner_mtype=distance_mtype)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame of mtype pd-multiindex or pd_multiindex_hier
        y : ignored, present only for interface compatibility
        """
        # store full data as indexed X
        self._X = X

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame of mtype pd-multiindex or pd_multiindex_hier
        y : ignored, present only for interface compatibility

        Returns
        -------
        transformed version of X
        """
        distance = self.distance

        X_train = self._X

        distmat = distance(X, X_train)

        X_train_ind = X_train.index.droplevel(-1)
        X_ind = X.index.droplevel(-1)

        Xt = pd.DataFrame(distmat, columns=X_train_ind, index=X_ind)

        return Xt
