# -*- coding: utf-8 -*-
"""Channel Selection techniques for Multivariate Time Series Classification.

A transformer that selects a subset of channels/dimensions for time series
classification using a scoring system with an elbow point method.
"""

__author__ = ["haskarb", "a-pasos-ruiz", "TonyBagnall", "fkiraly"]
__all__ = ["ElbowClassSum", "ElbowClassPairwise"]


import itertools
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestCentroid

from sktime.datatypes import convert
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from sktime.transformations.base import BaseTransformer


def _eu_dist(x, y):
    """Calculate the euclidean distance."""
    return euclidean(x, y)


def _detect_knee_point(values, indices):
    """Find elbow point."""
    n_points = len(values)
    all_coords = np.vstack((range(n_points), values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    knee_idx = np.argmax(dist_to_line)
    knee = values[knee_idx]
    best_dims = [idx for (elem, idx) in zip(values, indices) if elem > knee]
    if len(best_dims) == 0:
        return [knee_idx], knee_idx

    return (best_dims,)


class _distance_matrix:
    """Create distance matrix."""

    def distance(self, centroid_frame):
        """Fuction to create DM."""
        distance_pair = list(
            itertools.combinations(range(0, centroid_frame.shape[0]), 2)
        )
        # exit()

        map_cls = centroid_frame.class_vals.to_dict()
        distance_frame = pd.DataFrame()
        for class_ in distance_pair:

            class_pair = []
            # calculate the distance of centroid here
            for _, (q, t) in enumerate(
                zip(
                    centroid_frame.drop(["class_vals"], axis=1).iloc[class_[0], :],
                    centroid_frame.iloc[class_[1], :],
                )
            ):
                # print(eu_dist(q.values, t.values))
                class_pair.append(_eu_dist(q.values, t.values))
                dict_ = {
                    f"Centroid_{map_cls[class_[0]]}_{map_cls[class_[1]]}": class_pair
                }
                # print(class_[0])

            distance_frame = pd.concat([distance_frame, pd.DataFrame(dict_)], axis=1)

        return distance_frame


class _shrunk_centroid:
    """Create centroid."""

    def __init__(self, shrink=0):
        self.shrink = shrink

    def create_centroid(self, X, y):
        """Create the centroid for each class."""
        _, ncols, _ = X.shape
        cols = ["dim_" + str(i) for i in range(ncols)]
        ts = X
        centroids = []

        # le = LabelEncoder()
        # y_ind = le.fit_transform(y)

        for dim in range(ts.shape[1]):
            train = ts[:, dim, :]
            clf = NearestCentroid(train)
            clf = NearestCentroid(shrink_threshold=self.shrink)
            clf.fit(train, y)
            centroids.append(clf.centroids_)

        centroid_frame = from_3d_numpy_to_nested(
            np.stack(centroids, axis=1), column_names=cols
        )
        centroid_frame["class_vals"] = clf.classes_

        return centroid_frame.reset_index(drop=True)


class ElbowClassSum(BaseTransformer):
    """Elbow Class Sum (ECS) transformer to select a subset of channels/variables.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class centroid. The
    ECS selects the subset of channels using the elbow method, which maximizes the
    distance between the class centroids by aggregating the distance for every
    class pair across each channel.

    Note: Channels, variables, dimensions, features are used interchangeably in
    literature. E.g., channel selection = variable selection.

    Parameters
    ----------
    distance: sktime pairwise panel transform, str, or callable, optional, default=None
        if panel transform, will be used directly as the distance in the algorithm
        default None = euclidean distance on flattened series, FlatDist(ScipyDist())
        if str, will behave as FlatDist(ScipyDist(distance)) = scipy dist on flat series
        if callable, must be univariate nested_univ x nested_univ -> 2D float np.array

    Attributes
    ----------
    channels_selected_ : list of integer
        List of variables/channels selected by the estimator
        integers (iloc reference), referring to variables/channels by order
    channels_selected_idx_ : list of pandas compatible index elements
        List of variables/channels selected by the estimator
        if data are index-less (no channel/var names), identical to channels_selected
    distance_frame_ : DataFrame
        distance matrix of the class centroids pair and channels.
            ``shape = [n_channels, n_class_centroids_pairs]``
        Table 1 provides an illustration in [1].
    train_time_ : int
        Time taken to train the ECS.

    Notes
    -----
    Original repository: https://github.com/mlgig/Channel-Selection-MTSC

    References
    ----------
    ..[1]: Bhaskar Dhariyal et al. “Fast Channel Selection for Scalable Multivariate
    Time Series Classification.” AALTD, ECML-PKDD, Springer, 2021

    Examples
    --------
    >>> from sktime.transformations.panel.channel_selection import ElbowClassSum
    >>> from sktime.utils._testing.panel import make_classification_problem
    >>> X, y = make_classification_problem(n_columns=3, n_classes=3, random_state=42)
    >>> cs = ElbowClassSum()
    >>> cs.fit(X, y)
    ElbowClassSum(...)
    >>> Xt = cs.transform(X)

    Any sktime compatible distance can be used, e.g., DTW distance:
    >>> from sktime.dists_kernels import DtwDist
    >>>
    >>> cs = ElbowClassSum(distance=DtwDist())
    >>> cs.fit(X, y)
    ElbowClassSum(...)
    >>> Xt = cs.transform(X)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        # "scitype:transform-output": "Primitives",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "numpy1D",  # which mtypes do _fit/_predict support for y?
        "requires_y": True,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
    }

    def __init__(self, distance=None):

        self.distance = distance

        super(ElbowClassSum, self).__init__()

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

    def _fit(self, X, y):
        """Fit ECS to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : reference to self.
        """
        t = self.distance_

        start = int(round(time.time() * 1000))
        centroid_obj = _shrunk_centroid(0)

        X_np = convert(X, "nested_univ", "numpy3D")
        centroids = centroid_obj.create_centroid(X_np, y)
        centroids_no_y = centroids.drop("class_vals", axis=1)
        centroids_no_y.columns = X.columns

        dists = [t(X[[c]]).sum() for c in X.columns]
        dists = pd.Series(dists)
        self.distance_frame_ = dists

        distance = dists.sort_values(ascending=False).values
        indices = dists.sort_values(ascending=False).index

        idx = _detect_knee_point(distance, indices)[0]

        self.channels_selected_ = idx
        self.channels_selected_idx_ = [X.columns[i] for i in idx]

        self.train_time_ = int(round(time.time() * 1000)) - start

        return self

    def _transform(self, X, y=None):
        """
        Transform X and return a transformed version.

        Parameters
        ----------
        X : pandas DataFrame or np.ndarray
            The input data to transform.

        Returns
        -------
        output : pandas DataFrame
            X with a subset of channels
        """
        return X[self.channels_selected_idx_]

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.dists_kernels import DtwDist

        # default params
        params1 = {}

        # with custom distance
        params2 = {"distance": DtwDist()}

        # with string shorthand
        params3 = {"distance": "cosine"}

        return [params1, params2, params3]


class ElbowClassPairwise(BaseTransformer):
    """Elbow Class Pairwise (ECP) transformer to select a subset of channels.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class centroid. The ECP
    selects the subset of channels using the elbow method that maximizes the
    distance between each class centroids pair across all channels.

    Note: Channels, variables, dimensions, features are used interchangeably in
    literature.

    Attributes
    ----------
    channels_selected_ : list of integers; integer being the index of the channel
        List of channels selected by the ECS.
    distance_frame_ : DataFrame
        distance matrix of the class centroids pair and channels.
            ``shape = [n_channels, n_class_centroids_pairs]``
        Table 1 provides an illustration in [1].
    train_time_ : int
        Time taken to train the ECP.

    Notes
    -----
    Original repository: https://github.com/mlgig/Channel-Selection-MTSC

    References
    ----------
    ..[1]: Bhaskar Dhariyal et al. “Fast Channel Selection for Scalable Multivariate
    Time Series Classification.” AALTD, ECML-PKDD, Springer, 2021

    Examples
    --------
    >>> from sktime.transformations.panel.channel_selection import ElbowClassPairwise
    >>> from sktime.utils._testing.panel import make_classification_problem
    >>> X, y = make_classification_problem(n_columns=3, n_classes=3, random_state=42)
    >>> cs = ElbowClassPairwise()
    >>> cs.fit(X, y)
    ElbowClassPairwise(...)
    >>> Xt = cs.transform(X)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        # "scitype:transform-output": "Primitives",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "numpy1D",  # which mtypes do _fit/_predict support for y?
        "requires_y": True,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
    }

    def __init__(self):
        super(ElbowClassPairwise, self).__init__()

    def _fit(self, X, y):
        """Fit ECP to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : reference to self.

        """
        self.channels_selected_ = []
        start = int(round(time.time() * 1000))
        centroid_obj = _shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(), y)
        obj = _distance_matrix()
        self.distance_frame_ = obj.distance(df)

        for pairdistance in self.distance_frame_.iteritems():
            distance = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index

            self.channels_selected_.extend(_detect_knee_point(distance, indices)[0])
            self.channels_selected_ = list(set(self.channels_selected_))
        self.train_time_ = int(round(time.time() * 1000)) - start

        return self

    def _transform(self, X, y=None):
        """
        Transform X and return a transformed version.

        Parameters
        ----------
        X : pandas DataFrame or np.ndarray
            The input data to transform.

        Returns
        -------
        output : pandas DataFrame
            X with a subset of channels
        """
        return X[:, self.channels_selected_, :]
