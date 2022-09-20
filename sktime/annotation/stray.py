# -*- coding: utf-8 -*-
"""Tests for STRAY (Search TRace AnomalY) outlier estimator."""

import warnings
from typing import Dict

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

from sktime.transformations.base import BaseTransformer

__author__ = ["KatieBuc"]
__all__ = ["STRAY"]


class STRAY(BaseTransformer):
    """STRAY: robust anomaly detection in data streams with concept drift.

    This is based on STRAY (Search TRace AnomalY) _[1], which is a modification
    of HDoutliers _[2]. HDoutliers is a powerful algorithm for the detection of
    anomalous observations in a dataset, which has (among other advantages) the
    ability to detect clusters of outliers in multi-dimensional data without
    requiring a model of the typical behavior of the system. However, it suffers
    from some limitations that affect its accuracy. STRAY is an extension of
    HDoutliers that uses extreme value theory for the anomolous threshold
    calculation, to deal with data streams that exhibit non-stationary behavior.

    Parameters
    ----------
    alpha : float, optional (default=0.01)
        Threshold for determining the cutoff for outliers. Observations are
        considered outliers if they fall in the (1 - alpha) tail of
        the distribution of the nearest-neighbor distances between exemplars.
    k : int, optional (default=10)
        Number of neighbours considered.
    knn_algorithm : str {"auto", "ball_tree", "kd_tree", "brute"}, optional
        (default="brute")
        Algorithm used to compute the nearest neighbors, from
        sklearn.neighbors.NearestNeighbors
    p : float, optional (default=0.5)
        Proportion of possible candidates for outliers. This defines the starting point
        for the bottom up searching algorithm.
    size_threshold : int, optional (default=50)
        Sample size to calculate an emperical threshold.
    outlier_tail : str {"min", "max"}, optional (default="max")
        Direction of the outlier tail.

    Attributes
    ----------
    score_ : pd.Series
        Outlier score of each data point in X.
    y_ : pd.Series
        Outlier boolean flag for each data point in X.

    References
    ----------
    .. [1] Talagala, Priyanga Dilini, Rob J. Hyndman, and Kate Smith-Miles.
    "Anomaly detection in high-dimensional data." Journal of Computational
    and Graphical Statistics 30.2 (2021): 360-374.
    .. [2] Wilkinson, Leland. "Visualizing big data outliers through
    distributed aggregation." IEEE transactions on visualization and
    computer graphics 24.1 (2017): 256-266.

    Examples
    --------
    >>> from sktime.annotation.stray import STRAY
    >>> from sktime.datasets import load_airline
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> import numpy as np
    >>> X = load_airline().to_frame().to_numpy()
    >>> scaler = MinMaxScaler()
    >>> X = scaler.fit_transform(X)
    >>> model = STRAY(k=3)
    >>> y = model.fit_transform(X)
    >>> y[:5]
    array([False, False, False, False, False])
    """

    _tags = {
        "handles-missing-data": True,
        "X_inner_mtype": "np.ndarray",
        "fit_is_empty": False,
        "skip-inverse-transform": True,
    }

    def __init__(
        self,
        alpha: float = 0.01,
        k: int = 10,
        knn_algorithm: str = "brute",
        p: float = 0.5,
        size_threshold: int = 50,
        outlier_tail: str = "max",
    ):
        self.alpha = alpha
        self.k = k
        self.knn_algorithm = knn_algorithm
        self.p = p
        self.size_threshold = size_threshold
        self.outlier_tail = outlier_tail
        super(STRAY, self).__init__()

    def _find_threshold(self, outlier_score: npt.ArrayLike, n: int) -> npt.ArrayLike:
        """Find Outlier Threshold.

        Parameters
        ----------
        outlier_score : np.ArrayLike
            The outlier scores determined by k nearast neighbours distance
        n : int
            The number of rows remaining in X when NA's are removed.

        Returns
        -------
        array of indices of the observations determined to be outliers.
        """
        if self.outlier_tail == "min":
            outlier_score = -outlier_score

        order = np.argsort(outlier_score)
        gaps = np.append(0, np.diff(outlier_score[order]))
        n4 = int(max(min(self.size_threshold, np.floor(n / 4)), 2))

        J = np.array(range(2, n4 + 1))
        start = int(max(np.floor(n * (1 - self.p)), 1))

        ghat = [
            0.0 if i < start else sum((J / (n4 - 1)) * gaps[i - J + 1])
            for i in range(n)
        ]

        log_alpha = np.log(1 / self.alpha)
        bound = np.Inf

        for i in range(start, n):
            if gaps[i] > log_alpha * ghat[i]:
                bound = outlier_score[order][i - 1]
                break

        return np.where(outlier_score > bound)[0]

    def _find_outliers_kNN(self, X: npt.ArrayLike, n: int) -> Dict:
        """Find outliers using kNN distance with maximum gap.

        Parameters
        ----------
        X : np.ArrayLike
            Data for anomaly detection (time series).
        n : int
            The number of rows remaining in X when NA's are removed.

        Returns
        -------
        dict of index of outliers and the outlier scores
        """
        if len(X.shape) == 1:
            nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(X.reshape(-1, 1))
            distances, _ = nbrs.kneighbors(X.reshape(-1, 1))
        else:
            nbrs = NearestNeighbors(
                n_neighbors=n if self.k >= n else self.k + 1,
                algorithm=self.knn_algorithm,
            ).fit(X)
            distances, _ = nbrs.kneighbors(X)

        if self.k == 1:
            d = distances[:, 1]
        else:
            diff = np.apply_along_axis(np.diff, 1, distances)
            d = distances[range(n), np.apply_along_axis(np.argmax, 1, diff) + 1]

        out_index = self._find_threshold(d, n)
        return {"idx_outliers": out_index, "out_scores": d}

    def _find_outliers(self, X):
        """Detect Anomalies in High Dimensional Data.

        Parameters
        ----------
        X : np.ArrayLike
            Data for anomaly detection (time series).

        Returns
        -------
        dict of anomalies and their corresponding scores
        """
        r = np.shape(X)[0]
        idx_dropna = np.array([i for i in range(r) if not np.isnan(X[i]).any()])
        X_dropna = X[
            idx_dropna,
        ]

        n = np.shape(X_dropna)[0]
        outliers = self._find_outliers_kNN(X_dropna, n)

        # adjusted back to length r, for missing data
        slice_ = [True if i in outliers["idx_outliers"] else False for i in range(n)]
        idx_outliers = idx_dropna[slice_]  # index values from 1:r
        outlier_bool = np.array([1 if i in idx_outliers else 0 for i in range(r)])

        list_scores = outliers["out_scores"].tolist()
        outlier_scores = np.array(
            [list_scores.pop(0) if i in idx_dropna else np.nan for i in range(r)]
        )

        return {
            "outlier_scores": outlier_scores,
            "outlier_bool": outlier_bool,
        }

    def _fit(self, X: npt.ArrayLike, y: npt.ArrayLike = None):
        """Find outliers using STRAY (Search TRace AnomalY).

        Parameters
        ----------
        X : np.ArrayLike
            Data for anomaly detection (time series).
        y : pd.Series, optional
            Not used for this unsupervsed method.

        Returns
        -------
        self :
            Reference to self.
        """
        # remember X for transform
        self._X = X

        info_dict = self._find_outliers(X)
        self.score_ = info_dict["outlier_scores"]
        self.y_ = info_dict["outlier_bool"]

        return self

    def _transform(self, X: npt.ArrayLike, y: npt.ArrayLike = None) -> npt.ArrayLike:
        """Return anomaly detection.

        Parameters
        ----------
        X : np.ArrayLike
            Data for anomaly detection (time series).

        Returns
        -------
        y_ : np.ArrayLike
            Anomaly detection, boolean.
        """
        # fit again if data is different to fit, but don't store anything
        if not np.allclose(X, self._X, equal_nan=True):
            new_obj = STRAY(
                alpha=self.alpha,
                k=self.k,
                knn_algorithm=self.knn_algorithm,
                p=self.p,
                size_threshold=self.size_threshold,
                outlier_tail=self.outlier_tail,
            ).fit(X)
            warnings.warn(
                "Warning: Input data X differs from that given to fit(). "
                "Refitting with new input data, not storing updated public class "
                "attributes. For this, explicitly use fit(X) or fit_transform(X)."
            )
            return new_obj.y_.astype(bool)

        return self.y_.astype(bool)
