"""Compositors that use pairwise transformers for an ordinary transformstion."""

__author__ = ["fkiraly"]
__all__ = ["DistanceFeatures"]

import numpy as np
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

        X_requires_mapping = X.index.nlevels > 2
        X_train_requires_mapping = X_train.index.nlevels > 2

        if X_requires_mapping:
            # Convert hierachical index to panel index as distance expects panel data
            output_idx_names = X.index.names[:-1]
            new_index, X_mapping = self._map_hier_idx_to_panel_idx(X.index)
            X = X.set_index(new_index)

        if X_train_requires_mapping:
            # Convert hierachical index to panel index as distance expects panel data
            output_col_names = X_train.index.names[:-1]
            new_index, X_train_mapping = self._map_hier_idx_to_panel_idx(X_train.index)
            X_train = X_train.set_index(new_index)

        X_train_ind = X_train.index.droplevel(-1).unique()
        X_ind = X.index.droplevel(-1).unique()

        distmat = distance(X, X_train)

        Xt = pd.DataFrame(distmat, columns=X_train_ind, index=X_ind)

        if X_requires_mapping:
            # Convert the dummy index back to the expected hierachical index
            new_index = self._map_primitive_panel_idx_to_hier_idx(
                Xt.index, X_mapping, output_idx_names
            )
            if self.flatten_hierarchy and new_index.nlevels > 1:
                flatten_multiindex(new_index)
            Xt = Xt.set_index(new_index)

        if X_train_requires_mapping:
            # Convert the dummy columns back to the expected hierachical columns
            new_index = self._map_primitive_panel_idx_to_hier_idx(
                Xt.columns, X_train_mapping, output_col_names
            )
            if self.flatten_hierarchy and new_index.nlevels > 1:
                flatten_multiindex(new_index)
            Xt.columns = new_index

        return Xt

    def _map_hier_idx_to_panel_idx(self, idx):
        """Map hierarchical index to panel format."""
        unique_series = idx.droplevel(-1).unique()
        new_index = np.arange(len(unique_series))
        # Create a mapping
        mapping = dict(zip(unique_series, new_index))
        # Replace levels 0 up to -2 with new index
        new_index = pd.MultiIndex.from_tuples(
            [(mapping[x[:-1]], x[-1]) for x in idx.to_list()],
            names=["temp_index", idx.names[-1]],
        )

        return new_index, mapping

    def _map_primitive_panel_idx_to_hier_idx(self, idx, idx_mapping, idx_names):
        """Convert primitive index to hierarchical index."""
        idx_mapping_inv = {v: k for k, v in idx_mapping.items()}

        new_index = pd.MultiIndex.from_tuples(
            [idx_mapping_inv[x] for x in idx.to_list()],
            names=idx_names,
        )

        return new_index

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.dists_kernels.edit_dist import EditDist

        param1 = {}
        param2 = {"distance": "cityblock", "flatten_hierarchy": True}
        param3 = {"distance": EditDist.create_test_instance()}
        return [param1, param2, param3]
