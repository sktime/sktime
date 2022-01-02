# -*- coding: utf-8 -*-
"""Channel Seletion techniques for Multivariate Time Series Classification."""

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize

from sktime.datatypes._panel._convert import (
    from_3d_numpy_to_nested,
    from_nested_to_3d_numpy,
)


def eu_dist(x, y):
    """Calculate the euclidean distance."""
    return np.sqrt(np.sum((x - y) ** 2))


def detect_knee_point(values, indices):
    """Find elbow point."""
    n_points = len(values)
    all_coords = np.vstack((range(n_points), values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point
    scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    knee_idx = np.argmax(dist_to_line)
    knee = values[knee_idx]
    best_dims = [idx for (elem, idx) in zip(values, indices) if elem > knee]
    if len(best_dims) == 0:
        return [knee_idx], knee_idx

    return (best_dims,)


class distance_matrix:
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
                class_pair.append(eu_dist(q.values, t.values))
                dict_ = {
                    f"Centroid_{map_cls[class_[0]]}_{map_cls[class_[1]]}": class_pair
                }
                # print(class_[0])

            distance_frame = pd.concat([distance_frame, pd.DataFrame(dict_)], axis=1)

        return distance_frame


class shrunk_centroid:
    """Create centroid."""

    def __init__(self, shrink=0):
        self.shrink = shrink

    def create_centroid(self, X, y):
        """Create the centroid for each class."""
        # y = X.class_vals
        # X.drop('class_vals', axis = 1, inplace = True)
        cols = X.columns.to_list()
        ts = from_nested_to_3d_numpy(X)  # Contains TS in numpy format
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


class ecs(TransformerMixin, BaseEstimator):
    """Channel Selection Method: ECS."""

    def fit(self, X, y):
        """Convert training data."""
        centroid_obj = shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(), y)
        obj = distance_matrix()
        self.distance_frame = obj.distance(df)

        self.relevant_dims = []
        distance = self.distance_frame.sum(axis=1).sort_values(ascending=False).values
        indices = self.distance_frame.sum(axis=1).sort_values(ascending=False).index
        self.relevant_dims.extend(detect_knee_point(distance, indices)[0])

        return self

    def transform(self, X):
        """Return the transformed data."""
        return X.iloc[:, self.relevant_dims]


class kmeans(TransformerMixin, BaseEstimator):
    """Channel Selection Method: KMeans."""

    def fit(self, X, y):
        """Convert training data."""
        centroid_obj = shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(), y)
        obj = distance_matrix()
        self.distance_frame = obj.distance(df)
        # l2 normalisng for kmeans
        self.distance_frame = pd.DataFrame(
            normalize(self.distance_frame, axis=0),
            columns=self.distance_frame.columns.tolist(),
        )

        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(self.distance_frame)
        # Find the cluster name with maximum avg distance
        self.cluster = np.argmax(self.kmeans.cluster_centers_.mean(axis=1))
        self.relevant_dims = [
            id_ for id_, item in enumerate(self.kmeans.labels_) if item == self.cluster
        ]

        return self

    def transform(self, X):
        """Return the transformed data."""
        return X.iloc[:, self.relevant_dims]


class ecp(TransformerMixin, BaseEstimator):
    """Channel Selection Method: ECP."""

    def fit(self, X, y):
        """Convert training data."""
        centroid_obj = shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(), y)
        obj = distance_matrix()
        self.distance_frame = obj.distance(df)

        self.relevant_dims = []
        for pairdistance in self.distance_frame.iteritems():
            distance = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            self.relevant_dims.extend(detect_knee_point(distance, indices)[0])
            self.relevant_dims = list(set(self.relevant_dims))
        return self

    def transform(self, X):
        """Return the transformed data."""
        return X.iloc[:, self.relevant_dims]
