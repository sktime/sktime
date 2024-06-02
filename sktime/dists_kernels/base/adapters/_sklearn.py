# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for sklearn distances."""

__author__ = ["fkiraly", "Z-Fran"]
__all__ = ["_SklearnKnnAdapter"]

import numpy as np
import pandas as pd

from sktime.datatypes import convert


class _SklearnKnnAdapter:
    """Base adapter mixin for sklearn distances of KNeighbors."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "Z-Fran"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
    }

    def _one_element_distance_npdist(self, x, y, n_vars=None):
        if n_vars is None:
            n_vars = self.n_vars_
        x = np.reshape(x, (1, n_vars, -1))
        y = np.reshape(y, (1, n_vars, -1))
        return self._distance(x, y)[0, 0]

    def _one_element_distance_sktime_dist(self, x, y, n_vars=None):
        if n_vars is None:
            n_vars = self.n_vars_
        if n_vars == 1:
            x = np.reshape(x, (1, n_vars, -1))
            y = np.reshape(y, (1, n_vars, -1))
        elif self._X_metadata["is_equal_length"]:
            x = np.reshape(x, (-1, n_vars))
            y = np.reshape(y, (-1, n_vars))
            x_ix = pd.MultiIndex.from_product([[0], range(len(x))])
            y_ix = pd.MultiIndex.from_product([[0], range(len(y))])
            x = pd.DataFrame(x, index=x_ix)
            y = pd.DataFrame(y, index=y_ix)
        else:  # multivariate, unequal length
            # in _convert_X_to_sklearn, we have encoded the length as the first column
            # this was coerced to float, so we round to avoid rounding errors
            x_len = round(x[0])
            y_len = round(y[0])
            # pd.pivot switches the axes, compared to numpy
            x = np.reshape(x[1:], (n_vars, -1)).T
            y = np.reshape(y[1:], (n_vars, -1)).T
            # cut to length
            x = x[:x_len]
            y = y[:y_len]
            x_ix = pd.MultiIndex.from_product([[0], range(x_len)])
            y_ix = pd.MultiIndex.from_product([[0], range(y_len)])
            x = pd.DataFrame(x, index=x_ix)
            y = pd.DataFrame(y, index=y_ix)
        return self._distance(x, y)[0, 0]

    def _convert_X_to_sklearn(self, X):
        """Convert X to 2D numpy for sklearn."""
        # special treatment for unequal length series
        if not self._X_metadata["is_equal_length"]:
            # then we know we are dealing with pd-multiindex
            # as a trick to deal with unequal length data,
            # we flatten encode the length as the first column
            X_w_ix = X.reset_index(-1)
            X_pivot = X_w_ix.pivot(columns=[X_w_ix.columns[0]])
            # fillna since this creates nan but sklearn does not accept these
            # the fill value does not matter as the distance ignores it
            X_pivot = X_pivot.fillna(0).to_numpy()
            X_lens = X.groupby(X_w_ix.index).size().to_numpy()
            # add the first column, encoding length of individual series
            X_w_lens = np.concatenate([X_lens[:, None], X_pivot], axis=1)
            return X_w_lens

        # equal length series case
        if isinstance(X, np.ndarray):
            X_mtype = "numpy3D"
        else:
            X_mtype = "pd-multiindex"
        return convert(X, from_type=X_mtype, to_type="numpyflat")
