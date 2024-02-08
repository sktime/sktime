"""Compositors that use pairwise transformers for an ordinary transformstion."""

__author__ = ["fkiraly"]
__all__ = ["DistanceFeatures"]

import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.multiindex import flatten_multiindex


class DistanceFeatures(BaseTransformer):
    """Use distances to training series as features.

    In `transform`, returns tabular features as follows:
    for `i`-th series in `X`, returns all distances to series seen in `fit`
    `j` th column of `i`-th row is distance between `i`-th series in `transform`,
    and `j`-the series in `fit`. Column index is instance index in `fit`.
    If `fit` series was `Hierarchical`, hierarchy index is preserved.

    Parameters
    ----------
    distance: sktime pairwise panel transform, str, or callable, optional, default=None
        if panel transform, will be used directly as the distance in the algorithm
        default None = euclidean distance on flattened series, FlatDist(ScipyDist())
        if str, will behave as FlatDist(ScipyDist(distance)) = scipy dist on flat series
        if callable, must be distance_mtype x distance_mtype -> 2D float np.array
    distance_mtype : str, or list of str optional. default = None.
        mtype that distance expects for X and X2, if a callable
        only set this if distance is not BasePairwiseTransformerPanel descendant
    flatten_hierarchy : bool, optional, default=False.
        whether column hierarchy in `transform` return is flattened (using `__` concat),
        in case of a hierarchical series index seen in `fit`.

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.transformations.panel.compose_distance import DistanceFeatures
    >>> X_train, _ = load_unit_test(return_X_y=True, split="train")
    >>> X, _ = load_unit_test(return_X_y=True, split="test")
    >>> trafo = DistanceFeatures()
    >>> trafo.fit(X_train)
    DistanceFeatures(...)
    >>> Xt = trafo.transform(X)
    """

    _tags = {
        "authors": "fkiraly",
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
        # we leave remember_data as False, since updating self._X in update
        # would increase the number of columns in the transform return
        "remember_data": False,
    }

    def __init__(self, distance=None, distance_mtype=None, flatten_hierarchy=False):
        self.distance = distance
        self.distance_mtype = distance_mtype
        self.flatten_hierarchy = flatten_hierarchy

        super().__init__()

        from sktime.dists_kernels import (
            BasePairwiseTransformerPanel,
            FlatDist,
            ScipyDist,
        )

        if distance is None:
            self.distance_ = FlatDist(ScipyDist())
        elif isinstance(distance, str):
            self.distance_ = FlatDist(ScipyDist(metric=distance))
        elif isinstance(distance, BasePairwiseTransformerPanel):
            self.distance_ = distance.clone()
        else:
            self.distance_ = distance

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
        distance = self.distance_

        X_train = self._X

        X_train_ind = X_train.index.droplevel(-1).unique()
        X_ind = X.index.droplevel(-1).unique()

        def _coerce_to_panel(x):
            """Coerce hierarchical or pandel x to panel."""
            nlevels = x.index.nlevels
            if nlevels > 2:
                return x.droplevel(list(range(nlevels - 2)))
            else:
                return x

        X = _coerce_to_panel(X)
        X_train = _coerce_to_panel(X_train)

        distmat = distance(X, X_train)

        if self.flatten_hierarchy:
            X_ind = flatten_multiindex(X_ind)

        Xt = pd.DataFrame(distmat, columns=X_train_ind, index=X_ind)

        return Xt
