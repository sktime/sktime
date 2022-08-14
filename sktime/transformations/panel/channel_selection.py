# -*- coding: utf-8 -*-
"""Channel Selection techniques for Multivariate Time Series Classification.

A transformer that selects a subset of channels/dimensions for time series
classification using a scoring system with an elbow point method.
"""

__author__ = ["haskarb", "a-pasos-ruiz", "TonyBagnall"]
__all__ = ["ElbowClassSum", "ElbowClassPairwise"]


import itertools
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestCentroid

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
    """Elbow Class Sum (ECS) transformer to select a subset of channels.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class centroid. The
    ECS selects the subset of channels using the elbow method, which maximizes the
    distance between the class centroids by aggregating the distance for every
    class pair across each channel.


    Attributes
    ----------
    channels_selected_ : list
        List of channels selected by the ECS.
    distance_frame_ : DataFrame
        Distance matrix between the class centroids.
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
    >>> from sktime.datasets import load_UCR_UEA_dataset
    >>> cs = ElbowClassSum()
    >>> X_train, y_train = load_UCR_UEA_dataset(
    ...     "Cricket", split="train", return_X_y=True
    ... )
    >>> cs.fit(X_train, y_train)
    >>> Xt = cs.transform(X_train)
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

        super(ElbowClassSum, self).__init__()

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
        self.channels_selected_ = []
        start = int(round(time.time() * 1000))
        centroid_obj = _shrunk_centroid(0)
        centroids = centroid_obj.create_centroid(X.copy(), y)
        distances = _distance_matrix()
        self.distance_frame_ = distances.distance(centroids)
        distance = self.distance_frame_.sum(axis=1).sort_values(ascending=False).values
        indices = self.distance_frame_.sum(axis=1).sort_values(ascending=False).index
        self.channels_selected_.extend(_detect_knee_point(distance, indices)[0])
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


class ElbowClassPairwise(BaseTransformer):
    """Elbow Class Pairwise (ECP) transformer to select a subset of channels.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class centroid. The ECP
    selects the subset of channels using the elbow method that maximizes the
    distance between each class centroids pair across all channels.

    Attributes
    ----------
    channels_selected_ : list
        List of channels selected by the ECP.
    distance_frame_ : DataFrame
        Distance matrix between the class centroids.
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
    >>> from sktime.datasets import load_UCR_UEA_dataset
    >>> cs = ElbowClassPairwise()
    >>> X_train, y_train = load_UCR_UEA_dataset(
    ...     "Cricket", split="train", return_X_y=True
    ... )
    >>> cs.fit(X_train, y_train)
    >>> Xt = cs.transform(X_train)
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
        self._is_fitted = True
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
