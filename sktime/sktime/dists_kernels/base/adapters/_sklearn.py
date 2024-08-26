# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for sklearn distances."""

__author__ = ["fkiraly", "Z-Fran"]
__all__ = ["_SklearnDistanceAdapter"]

from inspect import signature

import numpy as np
import pandas as pd

from sktime.datatypes import convert


class _SklearnDistanceAdapter:
    """Adapter mixin to pass multivariate unequal length distance as 2D distance.

    Many sklearn compatible estimators, such as KNeighbors, can accept custom distances,
    but will expect callables that take 2D arrays as input.

    This mixin adapts time series distances to that interface,
    which can in-principle take multivariate or unequal length time series.

    The distance adapted is the parameter ``distance``.

    The pattern to use is:

    * use instances of this class internally, passed to the sklearn estimator.
      Instances are callable, compatible with ``sklearn`` estimators,
      of signature ``metric : (x: 1D np.ndarray, y: 1D np.ndarray) -> float``
    * adapt the sklearn estimator, and convert time series data to 2D numpy arrays
      via the ``_convert_X_to_sklearn`` method

    This way, the initial conversion in this distance, and ``_convert_X_to_sklearn``
    will cancel each other out, and distance or kernel based ``sklearn`` estimators
    written only for tabular data can be applied to time series data
    with ``sklearn`` compatible mtypes.

    To avoid repetitive checks,
    metadata of the time series must be passed to the adapter, as:

    * ``n_vars``: number of variables in the time series data
    * ``is_equal_length``: whether the time series data is of equal length

    If ``is_equal_length`` is True, the internal distance
    is simply the distance applied to time series flattened to 1D,
    and ``_convert_X_to_sklearn`` will flatten the time series data.

    If ``is_equal_length`` is False, the internal distance
    will have a leading scalar dimension encoding the length of the individual series,
    and ``_convert_X_to_sklearn`` will produce a flattened vector
    with the length encoded as the first column in addition.

    Parameters
    ----------
    distance : sklearn BasePairwiseTransformerPanel distance, or str
        Distance object or string code for distance.
        If string code, adapts one of the numba distances from ``sktime.distances``.
    distance_params : dict, optional
        Parameters to pass to the distance object.
        For BasePairwiseTransformerPanel distances, parameters should be
        directly passed as object parameters.
    n_vars : int, optional, default=1
        Number of variables in the time series data.
    is_equal_length : bool, optional, default=True
        Whether the time series data is of equal length.
    """

    def __init__(self, distance, distance_params=None, n_vars=1, is_equal_length=True):
        self.distance = distance
        self.distance_params = distance_params
        self.n_vars = n_vars
        self.is_equal_length = is_equal_length

    def __call__(self, x, y):
        """Compute distance - unified interface to str code and callable."""
        # sklearn wants distance callable element-wise,
        # numpy1D x numpy1D -> float
        # sktime distance classes are Panel x Panel -> numpy2D
        # and the numba distances are numpy3D x numpy3D -> numpy2D
        # so we need to wrap the sktime distances
        if isinstance(self.distance, str):
            # numba distances
            metric = self._one_element_distance_npdist
        else:
            # sktime distance classes
            metric = self._one_element_distance_sktime_dist
        return metric(x, y)

    def _distance(self, X, X2=None):
        """Compute distance - unified interface to str code and callable.

        If self.distance is a string, it is assumed to be a numba distance,
        and X, X2 are assumed in numpy3D format.

        If self.distance is a callable, it is assumed to be a sktime distance,
        and X, X2 are assumed in any of the sktime Panel formats,
        e.g., pd-multiindex, numpy3D.

        Consumers of this method should ensure that the input is in the correct format.

        This method should not be used as a direct public interface,
        only for internal use in estimators making use of this adapter.
        """
        distance = self.distance
        distance_params = self.distance_params
        if distance_params is None:
            distance_params = {}
        if isinstance(distance, str):
            from sktime.distances import pairwise_distance

            return pairwise_distance(X, X2, distance, **distance_params)
        else:
            if X2 is not None:
                return distance(X, X2, **distance_params)
            # if X2 is None, check if distance allows None X2 to mean "X2=X"
            else:
                sig = signature(distance).parameters
                X2_sig = sig[list(sig.keys())[1]]
                if X2_sig.default is not None:
                    return distance(X, X2, **distance_params)
                else:
                    return distance(X, **distance_params)

    def _one_element_distance_npdist(self, x, y, n_vars=None):
        if n_vars is None:
            n_vars = self.n_vars
        x = np.reshape(x, (1, n_vars, -1))
        y = np.reshape(y, (1, n_vars, -1))
        return self._distance(x, y)[0, 0]

    def _one_element_distance_sktime_dist(self, x, y, n_vars=None):
        if n_vars is None:
            n_vars = self.n_vars
        if n_vars == 1:
            x = np.reshape(x, (1, n_vars, -1))
            y = np.reshape(y, (1, n_vars, -1))
        elif self.is_equal_length:
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
        if not self.is_equal_length:
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
