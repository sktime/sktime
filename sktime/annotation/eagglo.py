"""E-Agglo: agglomerative clustering algorithm that preserves observation order."""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sktime.transformations.base import BaseTransformer

__author__ = ["KatieBuc"]
__all__ = ["EAgglo"]


class EAgglo(BaseTransformer):
    """Hierarchical agglomerative estimation of multiple change points.

    E-Agglo is a non-parametric clustering approach for multivariate timeseries[1]_,
    where neighboring segments are sequentially merged_ to maximize a goodness-of-fit
    statistic. Unlike most general purpose agglomerative clustering algorithms, this
    procedure preserves the time ordering of the observations.

    This method can detect distributional change within an independent sequence,
    and does not make any distributional assumptions (beyond the existence of an
    alpha-th moment). Estimation is performed in a manner that simultaneously
    identifies both the number and locations of change points.

    Parameters
    ----------
    member : array_like (default=None)
        Assigns points to to the initial cluster membership, therefore the first
        dimension should be the same as for data. If ``None`` it will be initialized
        to dummy vector where each point is assigned to separate cluster.
    alpha : float (default=1.0)
        Fixed constant alpha in (0, 2] used in the divergence measure, as the
        alpha-th absolute moment, see equation (4) in [1]_.
    penalty : str or callable or None (default=None)
        Function that defines a penalization of the sequence of goodness-of-fit
        statistic, when overfitting is a concern. If ``None`` not penalty is applied.
        Could also be an existing penalty name, either ``len_penalty`` or
        ``mean_diff_penalty``.

    Attributes
    ----------
    merged_ : array_like
        2D ``array_like`` outlining which clusters were merged_ at each step.
    gof_ : float
        goodness-of-fit statistic for current clsutering.
    cluster_ : array_like
        1D ``array_like`` specifying which cluster each row of input data
        X belongs to.

    Notes
    -----
    Based on the work from [1]_.

    - source code based on: https://github.com/cran/ecp/blob/master/R/e_agglomerative.R
    - paper available at: https://www.tandfonline.com/doi/full/10.1080/01621459.\
        2013.849605

    References
    ----------
    .. [1] Matteson, David S., and Nicholas A. James. "A nonparametric approach for
    multiple change point analysis of multivariate data." Journal of the American
    Statistical Association 109.505 (2014): 334-345.

    .. [2] James, Nicholas A., and David S. Matteson. "ecp: An R package for
    nonparametric multiple change point analysis of multivariate data." arXiv preprint
    arXiv:1309.3295 (2013).

    Examples
    --------
    >>> from sktime.annotation.datagen import piecewise_normal_multivariate
    >>> X = piecewise_normal_multivariate(means=[[1, 3], [4, 5]], lengths=[3, 4],
    ... random_state = 10)
    >>> from sktime.annotation.eagglo import EAgglo
    >>> model = EAgglo()
    >>> model.fit_transform(X)
    array([0, 0, 0, 1, 1, 1, 1])
    """

    _tags = {
        "fit_is_empty": False,
    }

    def __init__(
        self,
        member=None,
        alpha=1.0,
        penalty=None,
    ):
        self.member = member
        self.alpha = alpha
        self.penalty = penalty
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None):
        """Find optimally clustered segments.

        First, by determining which pairs of adjacent clusters will be merged_. Then,
        this process is repeated, recording the goodness-of-fit statistic at each step,
        until all observations belong to a single cluster. Finally, the estimated number
        of change points is estimated by the clustering that maximizes the goodness-of-
        fit statistic over the entire merging sequence.

        Parameters
        ----------
        X : pd.DataFrame
            Data for anomaly detection (time series).
        y : pd.Series, optional
            Not used for this unsupervsed method.

        Returns
        -------
        self :
            Reference to self.
        """
        self._X = X

        if self.alpha <= 0 or self.alpha > 2:
            raise ValueError(
                f"allowed values for 'alpha' are (0, 2], got: {self.alpha}"
            )

        self._initialize_params(X)

        # find which clusters optimize the gof_ and then update the distances
        for K in range(self.n_cluster - 1, 2 * self.n_cluster - 2):
            i, j = self._find_closest(K)
            self._update_distances(i, j, K)

        def filter_na(i):
            return list(
                filter(
                    lambda v: v == v,
                    self.progression[i,],
                )
            )

        # penalize the gof_ statistic
        if self.penalty is not None:
            penalty_func = self._get_penalty_func()
            cps = [filter_na(i) for i in range(len(self.progression))]
            self.gof_ += list(map(penalty_func, cps))

        # get the set of change points for the "best" clustering
        idx = np.argmax(self.gof_)
        self._estimates = np.sort(filter_na(idx))

        # remove change point N+1 if a cyclic merger was performed
        self._estimates = (
            self._estimates[:-1] if self._estimates[0] != 0 else self._estimates
        )

        # create final membership vector
        def get_cluster(estimates):
            return np.repeat(
                range(len(np.diff(estimates))), np.diff(estimates).astype(int)
            )

        if self._estimates[0] == 0:
            self.cluster_ = get_cluster(self._estimates)
        else:
            tmp = get_cluster(np.append([0], self._estimates))
            self.cluster_ = np.append(tmp, np.zeros(X.shape[0] - len(tmp)))

        return self

    def _transform(self, X: pd.DataFrame, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series of mtype X_inner_mtype
            Data to be transformed
        y : Series of mtype y_inner_mtype, default=None
            Not required for this unsupervised transform.

        Returns
        -------
        cluster
            numeric representation of cluster membership for each row of X.
        """
        # fit again if indices not seen, but don't store anything
        if not X.index.equals(self._X.index):
            X_full = X.combine_first(self._X)
            new_eagglo = EAgglo(
                member=self.member,
                alpha=self.alpha,
                penalty=self.penalty,
            ).fit(X_full)
            warnings.warn(
                "Warning: Input data X differs from that given to fit(). "
                "Refitting with both the data in fit and new input data, not storing "
                "updated public class attributes. For this, explicitly use fit(X) or "
                "fit_transform(X).",
                stacklevel=2,
            )
            return new_eagglo.cluster_

        return self.cluster_

    def _initialize_params(self, X: pd.DataFrame) -> None:
        """Initialize parameters and store to self."""
        self._member = np.array(
            self.member if self.member is not None else range(X.shape[0])
        )

        unique_labels = np.sort(np.unique(self._member))
        self.n_cluster = len(unique_labels)

        # relabel clusters to be consecutive numbers (when user specified)
        for i in range(self.n_cluster):
            self._member[np.where(self._member == unique_labels[i])[0]] = i

        # check if sorted
        if not all(sorted(self._member) == self._member):
            raise ValueError("'_member' should be sorted")

        self.sizes = np.zeros(2 * self.n_cluster)
        self.sizes[: self.n_cluster] = [
            sum(self._member == i) for i in range(self.n_cluster)
        ]  # calculate initial cluster sizes

        # array of within distances
        grouped = X.copy().set_index(self._member).groupby(level=0)
        within = grouped.apply(lambda x: get_distance(x, x, self.alpha))

        # array of between-within distances
        self.distances = np.empty((2 * self.n_cluster, 2 * self.n_cluster))

        for i, xi in grouped:
            self.distances[: self.n_cluster, i] = (
                2 * grouped.apply(lambda xj: get_distance(xi, xj, self.alpha))
                - within[i]
                - within
            )

        np.fill_diagonal(self.distances, 0)

        # set up left and right neighbors
        # special case for clusters 0 and n_cluster-1 to allow for cyclic merging
        self.left = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.left[: self.n_cluster] = [
            i - 1 if i >= 1 else self.n_cluster - 1 for i in range(self.n_cluster)
        ]

        self.right = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.right[: self.n_cluster] = [
            i + 1 if i + 1 < self.n_cluster else 0 for i in range(self.n_cluster)
        ]

        # True means that a cluster has not been merged_
        self.open = np.ones(2 * self.n_cluster - 1, dtype=bool)

        # which clusters were merged_ at each step
        self.merged_ = np.empty((self.n_cluster - 1, 2))

        # set initial gof_ value
        self.gof_ = np.array(
            [
                sum(
                    self.distances[i, self.left[i]] + self.distances[i, self.right[i]]
                    for i in range(self.n_cluster)
                )
            ]
        )

        # change point progression
        self.progression = np.empty((self.n_cluster, self.n_cluster + 1))
        self.progression[0, :] = [
            sum(self.sizes[:i]) if i > 0 else 0 for i in range(self.n_cluster + 1)
        ]  # N + 1 for cyclic mergers

        # array to specify the starting point of a cluster
        self.lm = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.lm[: self.n_cluster] = range(self.n_cluster)

    def _gof_update(self, i: int) -> float:
        """Compute the updated goodness-of-fit statistic, left cluster given by i."""
        fit = self.gof_[-1]
        j = self.right[i]

        # get new left and right clusters
        rr = self.right[j]
        ll = self.left[i]

        # remove unneeded values in the gof_
        fit -= 2 * (
            self.distances[i, j] + self.distances[i, ll] + self.distances[j, rr]
        )

        # get cluster sizes
        n1 = self.sizes[i]
        n2 = self.sizes[j]

        # add distance to new left cluster
        n3 = self.sizes[ll]
        k = (
            (n1 + n3) * self.distances[i, ll]
            + (n2 + n3) * self.distances[j, ll]
            - n3 * self.distances[i, j]
        ) / (n1 + n2 + n3)
        fit += 2 * k

        # add distance to new right
        n3 = self.sizes[rr]
        k = (
            (n1 + n3) * self.distances[i, rr]
            + (n2 + n3) * self.distances[j, rr]
            - n3 * self.distances[i, j]
        ) / (n1 + n2 + n3)
        fit += 2 * k

        return fit

    def _find_closest(self, K: int) -> tuple[int, int]:
        """Determine which clusters will be merged_, for K clusters.

        Greedily optimize the goodness-of-fit statistic by merging the pair of adjacent
        clusters that results in the largest increase of the statistic's value.

        Parameters
        ----------
        K: int
            Number of clusters

        Returns
        -------
        result : Tuple[int, int]
            Tuple of left cluster and right cluster index values
        """
        best_fit = -1e10
        result = (0, 0)

        # iterate through each cluster to see how the gof_ value changes if merged_
        for i in range(K + 1):
            if self.open[i]:
                gof_ = self._gof_update(i)
                if gof_ > best_fit:
                    best_fit = gof_
                    result = (i, self.right[i])

        self.gof_ = np.append(self.gof_, best_fit)
        return result

    def _update_distances(self, i: int, j: int, K: int) -> None:
        """Update distance from new cluster to other clusters, store to self."""
        # which clusters were merged_, info only
        self.merged_[K - self.n_cluster + 1, 0] = (
            -i if i <= self.n_cluster else i - self.n_cluster
        )
        self.merged_[K - self.n_cluster + 1, 1] = (
            -j if j <= self.n_cluster else j - self.n_cluster
        )

        # update left and right neighbors
        ll = self.left[i]
        rr = self.right[j]
        self.left[K + 1] = ll
        self.right[K + 1] = rr
        self.right[ll] = K + 1
        self.left[rr] = K + 1

        # update information about which clusters have been merged_
        self.open[i] = False
        self.open[j] = False

        # assign size to newly created cluster
        n1 = self.sizes[i]
        n2 = self.sizes[j]
        self.sizes[K + 1] = n1 + n2

        # update set of change points
        self.progression[K - self.n_cluster + 2, :] = self.progression[
            K - self.n_cluster + 1,
        ]
        self.progression[K - self.n_cluster + 2, self.lm[j]] = np.nan
        self.lm[K + 1] = self.lm[i]

        # update distances
        for k in range(K + 1):
            if self.open[k]:
                n3 = self.sizes[k]
                n = n1 + n2 + n3
                val = (
                    (n - n2) * self.distances[i, k]
                    + (n - n1) * self.distances[j, k]
                    - n3 * self.distances[i, j]
                ) / n
                self.distances[K + 1, k] = val
                self.distances[k, K + 1] = val

    def _get_penalty_func(self) -> Callable:  # sourcery skip: raise-specific-error
        """Define penalty function given (possibly string) input."""
        PENALTIES = {"len_penalty": len_penalty, "mean_diff_penalty": mean_diff_penalty}

        if callable(self.penalty):
            return self.penalty

        elif isinstance(self.penalty, str):
            if self.penalty in PENALTIES:
                return PENALTIES[self.penalty]

        raise Exception(
            f"'penalty' must be callable or {PENALTIES.keys()}, got {self.penalty}"
        )

    @classmethod
    def get_test_params(cls) -> list[dict]:
        """Test parameters."""
        return [
            {"alpha": 1.0, "penalty": None},
            {"alpha": 2.0, "penalty": "len_penalty"},
        ]


def get_distance(X: pd.DataFrame, Y: pd.DataFrame, alpha: float) -> float:
    """Calculate within/between cluster distance."""
    return np.power(cdist(X, Y, "euclidean"), alpha).mean()


def len_penalty(x: pd.DataFrame) -> int:
    """Penalize goodness-of-fit statistic for number of change points."""
    return -len(x)


def mean_diff_penalty(x: pd.DataFrame) -> float:
    """Penalize goodness-of-fit statistic.

    Favors segmentations with larger sizes, while taking into consideration the size of
    the new segments.
    """
    return np.mean(np.diff(np.sort(x)))
