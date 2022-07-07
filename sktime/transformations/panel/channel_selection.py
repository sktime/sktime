# -*- coding: utf-8 -*-
"""Channel Selection techniques for Multivariate Time Series Classification.

A transformer that selects a subset of dimensions/channels for time series
classification using a scoring system with an elbow point method.
"""

__author__ = ["haskarb", "a-pasos-ruiz", "TonyBagnall"]
__all__ = ["ElbowChannelSelection"]


import itertools
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize

from sktime.datatypes._panel._convert import from_3d_numpy_to_nested
from sktime.transformations.base import BaseTransformer

# from sklearn.preprocessing import normalize


def _eu_dist(x, y):
    """Calculate the euclidean distance."""
    return np.sqrt(np.sum((x - y) ** 2))


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


class ElbowChannelSelection(BaseTransformer):
    """ElbowChannelSelection (ECS) transformer to select a subset of dimensions.

    Apply to a set of multivariate time series instances (referred to as a Panel),
    in order to select dimensions using a scoring then elbow method (more details to
    follow).

    Parameters
    ----------
    normalise       : boolean, optional (default=True)
    n_jobs          : int, optional (default=1)
    random_state    : boolean, optional (default=None)


    """

    _tags = {
        # "scitype:transform-input": "Series",
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
    }

    def __init__(self, normalise=True, n_jobs=1, random_state=None):
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.channels_selected = []
        super(ElbowChannelSelection, self).__init__()

    def _fit(self, X, y):
        """
        Fit ECS to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        start = int(round(time.time() * 1000))
        centroid_obj = _shrunk_centroid(0)
        centroids = centroid_obj.create_centroid(X.copy(), y)
        distances = _distance_matrix()
        self.distance_frame_ = distances.distance(centroids)
        distance = self.distance_frame_.sum(axis=1).sort_values(ascending=False).values
        indices = self.distance_frame_.sum(axis=1).sort_values(ascending=False).index
        self.channels_selected.extend(_detect_knee_point(distance, indices)[0])
        self.train_time_ = int(round(time.time() * 1000)) - start
        return self

    def _transform(self, X, y=None):
        """
        Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        X with a subset of dimensions
        """
        return X[:, self.channels_selected, :]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"normalise": True}
        return params


class ClusterChannelSelection(BaseTransformer):
    """Channel Selection Method: KMeans.

    Apply KMeans to the distance matrix derived and
    creates two clusters to identify useful channels.

    Parameters
    ----------
    normalise       : boolean, optional (default=True)
    n_jobs          : int, optional (default=1)
    random_state    : boolean, optional (default=None)
    """

    _tags = {
        # "scitype:transform-input": "Series",
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
    }

    def __init__(self, normalise=True, n_jobs=1, random_state=None):
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None
        self.channels_selected = []
        self._is_fitted = False
        self.train_time = 0
        super(ClusterChannelSelection, self).__init__()

    def _fit(self, X, y):
        """
        Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        start = int(round(time.time() * 1000))
        centroid_obj = _shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(), y)
        obj = _distance_matrix()
        self.distance_frame_ = obj.distance(df)
        # l2 normalisng for kmeans
        self.distance_frame_ = pd.DataFrame(
            normalize(self.distance_frame_, axis=0),
            columns=self.distance_frame_.columns.tolist(),
        )

        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(self.distance_frame_)
        # Find the cluster name with maximum avg distance
        self.cluster = np.argmax(self.kmeans.cluster_centers_.mean(axis=1))
        self.channels_selected = [
            id_ for id_, item in enumerate(self.kmeans.labels_) if item == self.cluster
        ]
        self.train_time = int(round(time.time() * 1000)) - start
        self._is_fitted = True

        return self

    def _transform(self, X, y):
        """
        Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        X with a subset of dimensions
        """
        return X[:, self.channels_selected, :]


class ElbowClassPairwise(BaseTransformer):
    """Channel Selection Method: ECP.

    Apply to a set of multivariate time series instances (referred to as a Panel),
    in order to select dimensions using a scoring then elbow method (more details to
    follow).

    Parameters
    ----------
    normalise       : boolean, optional (default=True)
    n_jobs          : int, optional (default=1)
    random_state    : boolean, optional (default=None)
    """

    _tags = {
        # "scitype:transform-input": "Series",
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
    }

    def __init__(self, normalise=True, n_jobs=1, random_state=None):
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None
        self.channels_selected = []
        self._is_fitted = False
        self.train_time_ = 0
        super(ElbowClassPairwise, self).__init__()

    def _fit(self, X, y):
        """
        Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        start = int(round(time.time() * 1000))
        centroid_obj = _shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(), y)
        obj = _distance_matrix()
        self.distance_frame_ = obj.distance(df)

        for pairdistance in self.distance_frame_.iteritems():
            distance = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            self.channels_selected.extend(_detect_knee_point(distance, indices)[0])
            self.channels_selected = list(set(self.channels_selected))
        self.train_time_ = int(round(time.time() * 1000)) - start
        self._is_fitted = True
        return self

    def _transform(self, X, y):
        """
        Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        X with a subset of dimensions
        """
        return X[:, self.channels_selected, :]
