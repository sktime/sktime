# -*- coding: utf-8 -*-
"""Channel Selection techniques for Multivariate Time Series Classification.

A transformer that selects a subset of channels/dimensions for time series
classification using a scoring system with an elbow point method.
"""

__author__ = ["haskarb", "a-pasos-ruiz", "TonyBagnall", "fkiraly"]
__all__ = ["ElbowClassSum", "ElbowClassPairwise"]


import itertools

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder

from sktime.datatypes._panel._convert import (
    from_3d_numpy_to_nested,
    from_nested_to_3d_numpy,
)
from sktime.distances import distance
from sktime.transformations.base import BaseTransformer


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

    def __init__(self, distance):
        self.distance_ = str(distance)

    def create_distance_matrix(self, cp_frame):
        """Create a distance matrix between class_prototypes."""
        distance_pair = list(itertools.combinations(range(0, cp_frame.shape[0]), 2))

        idx_class = cp_frame.class_vals.to_dict()
        distance_frame = pd.DataFrame()
        for cls_ in distance_pair:

            class_pair = []
            # calculate the distance of centroid here
            for _, (q, t) in enumerate(
                zip(
                    cp_frame.drop(["class_vals"], axis=1).iloc[cls_[0], :],
                    cp_frame.iloc[cls_[1], :],
                )
            ):
                class_pair.append(distance(q.values, t.values, metric=self.distance_))
                dict_ = {
                    f"Centroid_{idx_class[cls_[0]]}_{idx_class[cls_[1]]}": class_pair
                }
            distance_frame = pd.concat([distance_frame, pd.DataFrame(dict_)], axis=1)

        return distance_frame


def _clip(x, low_value, high_value):
    return np.clip(x, low_value, high_value)


class Class_Prototype:
    """Create centroid."""

    def __init__(self, cp: str = "mean", mean_centering: bool = False):
        self.cp = cp
        self.mean_centering = mean_centering

        assert self.cp in ["mean", "median", "mad"], "Class prototype not supported."

    def _mad_median(self, class_X, median=None):

        _mad = median_abs_deviation(class_X, axis=0)

        low_value = median - _mad * 0.50
        high_value = median + _mad * 0.50
        # clip = lambda x: np.clip(x, low_value, high_value)
        class_X = np.apply_along_axis(
            _clip, axis=1, arr=class_X, low_value=low_value, high_value=high_value
        )

        return np.mean(class_X, axis=0), class_X

    def _class_mad(self, X, y):

        classes_ = np.unique(y)

        channel_median = []
        for class_ in classes_:
            class_idx = np.where(
                y == class_
            )  # find the indexes of data point where particular class is located

            class_median = np.median(X[class_idx], axis=0)
            class_median = self._mad_median(X[class_idx], class_median)[0]
            channel_median.append(class_median)

        return np.array(channel_median)

    def _class_median(self, X, y):

        classes_ = np.unique(y)

        channel_median = []
        for class_ in classes_:
            class_idx = np.where(
                y == class_
            )  # find the indexes of data point where particular class is located
            class_median = np.median(X[class_idx], axis=0)
            channel_median.append(class_median)
        return np.array(channel_median)

    def _mean_centering(self, cp_):
        return cp_.subtract(cp_.mean())

    def create_class_prototype(self, X, y):
        """Create the class prototype for each class."""
        cols = X.columns.to_list()
        ts = from_nested_to_3d_numpy(X)  # Contains TS in numpy format
        centroids = []

        le = LabelEncoder()
        y_ind = le.fit_transform(y)

        for dim in range(ts.shape[1]):  # iterating over channels
            train = ts[:, dim, :]

            if self.cp == "mean":
                clf = NearestCentroid()
                clf.fit(train, y_ind)
                centroids.append(clf.centroids_)

            elif self.cp == "median":
                ch_median = self._class_median(train, y_ind)
                centroids.append(ch_median)

            elif self.cp == "mad":
                ch_mad = self._class_mad(train, y_ind)
                centroids.append(ch_mad)

        centroid_frame = from_3d_numpy_to_nested(
            np.stack(centroids, axis=1), column_names=cols
        )

        if self.mean_centering:
            centroid_frame = centroid_frame.applymap(self._mean_centering)

        centroid_frame["class_vals"] = le.classes_

        return centroid_frame.reset_index(drop=True)


class ElbowClassSum(BaseTransformer):
    """Elbow Class Sum (ECS) transformer to select a subset of channels/variables.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class prototype. The
    ECS selects the subset of channels using the elbow method, which maximizes the
    distance between the class centroids by aggregating the distance for every
    class pair across each channel.

    Note: Channels, variables, dimensions, features are used interchangeably in
    literature. E.g., channel selection = variable selection.

    Parameters
    ----------
    distance : str
        Distance metric to use for creating the class prototype.
        Default: 'euclidean'
    class_prototype : str
        Type of class prototype to use for representing a class.
        Default: 'mean'
    mean_centering : bool
        If True, mean centering is applied to the class prototype.
        Default: False


    Attributes
    ----------
    class_prototype_ : DataFrame
        Class prototype for each class.
    distance_frame_ : DataFrame
        Distance matrix for each class pair.
        ``shape = [n_channels, n_class_prototype_pairs]``
    channels_selected_idx : list
        List of selected channels.
    rank: list
        Rank of channels based on the distance between class prototypes.
    class_prototype_ : DataFrame
        Class prototype for each class.


    Notes
    -----
    Original repository:
    1. https://github.com/mlgig/Channel-Selection-MTSC
    2. https://github.com/mlgig/ChannelSelectionMTSC

    References
    ----------
    ..[1]: Bhaskar Dhariyal et al. “Fast Channel Selection for Scalable Multivariate
    Time Series Classification.” AALTD, ECML-PKDD, Springer, 2021
    ..[2]: Bhaskar Dhariyal et al. “Scalable Classifier-Agnostic Channel Selection
    for Multivariate Time Series Classification", DAMI, ECML, Springer, 2023

    Examples
    --------
    >>> from sktime.transformations.panel.channel_selection import ElbowClassSum
    >>> from sktime.utils._testing.panel import make_classification_problem
    >>> X, y = make_classification_problem(n_columns=3, n_classes=3, random_state=42)
    >>> cs = ElbowClassSum()
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

    def __init__(
        self,
        distance="euclidean",
        class_prototype="mean",
        mean_centering=False,
    ):
        self.distance = distance
        self.mean_centering = mean_centering
        self.class_prototype = class_prototype

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
        centroid_obj = Class_Prototype(
            cp=self.class_prototype,
            mean_centering=self.mean_centering,
        )
        self.class_prototype_ = centroid_obj.create_class_prototype(X.copy(), y)
        obj = _distance_matrix(distance=self.distance)
        self.distance_frame = obj.create_distance_matrix(self.class_prototype_.copy())
        self.channels_selected_idx = []
        distance = self.distance_frame.sum(axis=1).sort_values(ascending=False).values
        indices = self.distance_frame.sum(axis=1).sort_values(ascending=False).index

        self.channels_selected_idx.extend(_detect_knee_point(distance, indices)[0])
        self.rank = self.channels_selected_idx

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
        return X.iloc[:, self.channels_selected_idx]


class ElbowClassPairwise(BaseTransformer):
    """Elbow Class Pairwise (ECP) transformer to select a subset of channels.

    Overview: From the input of multivariate time series data, create a distance
    matrix [1] by calculating the distance between each class centroid. The ECP
    selects the subset of channels using the elbow method that maximizes the
    distance between each class centroids pair across all channels.

    Note: Channels, variables, dimensions, features are used interchangeably in
    literature.

    Parameters
    ----------
    distance : str
        Distance metric to use for creating the class prototype.
        Default: 'euclidean'
    class_prototype : str
        Type of class prototype to use for representing a class.
        Default: 'mean'
    mean_centering : bool
        If True, mean centering is applied to the class prototype.
        Default: False


    Attributes
    ----------
    class_prototype_ : DataFrame
        Class prototype for each class.
    distance_frame_ : DataFrame
        Distance matrix for each class pair.
        ``shape = [n_channels, n_class_prototype_pairs]``
    channels_selected_idx : list
        List of selected channels.
    rank: list
        Rank of channels based on the distance between class prototypes.
    class_prototype_ : DataFrame
        Class prototype for each class.

    Notes
    -----
    Original repository:
    1. https://github.com/mlgig/Channel-Selection-MTSC
    2. https://github.com/mlgig/ChannelSelectionMTSC

    References
    ----------
    ..[1]: Bhaskar Dhariyal et al. “Fast Channel Selection for Scalable Multivariate
    Time Series Classification.” AALTD, ECML-PKDD, Springer, 2021
    ..[2]: Bhaskar Dhariyal et al. “Scalable Classifier-Agnostic Channel Selection
    for Multivariate Time Series Classification", DAMI, ECML, Springer, 2023

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
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "numpy1D",  # which mtypes do _fit/_predict support for y?
        "requires_y": True,  # does y need to be passed in fit?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "capability:unequal_length": False,
        # can the transformer handle unequal length time series (if passed Panel)?
    }

    def __init__(
        self,
        distance="euclidean",
        class_prototype="mad",
        mean_centering=False,
    ):

        self.distance = distance
        self.class_prototype = class_prototype
        self.mean_centering = mean_centering

        super(ElbowClassPairwise, self).__init__()

    def _rank(self):
        all_index = self.distance_frame.sum(axis=1).sort_values(ascending=False).index
        series = self.distance_frame.sum(axis=1)
        series.drop(
            index=list(set(all_index) - set(self.channels_selected_idx)), inplace=True
        )
        return series.sort_values(ascending=False).index.tolist()

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
        centroid_obj = Class_Prototype(
            cp=self.class_prototype, mean_centering=self.mean_centering
        )
        self.class_prototype_ = centroid_obj.create_class_prototype(
            X.copy(), y
        )  # Centroid created here
        obj = _distance_matrix(distance=self.distance)
        self.distance_frame = obj.create_distance_matrix(
            self.class_prototype_.copy()
        )  # Distance matrix created here

        all_chs = np.empty(
            self.class_prototype_.shape[1] - 1
        )  # -1 for removing class columsn
        all_chs.fill(0)
        self.channels_selected_idx = []

        for pairdistance in self.distance_frame.iteritems():
            distance_ = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            chs_dis = _detect_knee_point(distance_, indices)
            chs = chs_dis[0]
            self.channels_selected_idx.extend(chs)

        self.rank = self._rank()
        self.channels_selected_idx = list(set(self.channels_selected_idx))
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
        return X.iloc[:, self.channels_selected_idx]
