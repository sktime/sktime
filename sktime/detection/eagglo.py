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
        Assigns points to the initial cluster membership, therefore the first
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
    >>> from sktime.detection.datagen import piecewise_normal_multivariate
    >>> X = piecewise_normal_multivariate(means=[[1, 3], [4, 5]], lengths=[3, 4],
    ... random_state = 10)
    >>> from sktime.detection.eagglo import EAgglo
    >>> model = EAgglo()
    >>> model.fit_transform(X)
    array([0, 0, 0, 1, 1, 1, 1])
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "KatieBuc",
        "maintainers": "KatieBuc",
        # estimator type
        # --------------
        "fit_is_empty": False,
        "capability:categorical_in_X": False,
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
            row = self.progression[i,]
            return row[~np.isnan(row)]

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
        n_cluster = len(unique_labels)
        self.n_cluster = n_cluster

        # relabel clusters to be consecutive numbers (when user specified)
        self._member = np.searchsorted(unique_labels, self._member)

        # check if sorted
        if np.any(np.diff(self._member) < 0):
            raise ValueError("'_member' should be sorted")

        self.sizes = np.zeros(2 * n_cluster)
        # calculate initial cluster sizes
        cluster_sizes = np.bincount(self._member, minlength=n_cluster)
        self.sizes[:n_cluster] = cluster_sizes

        # array of between-within distances
        self.distances = np.empty((2 * n_cluster, 2 * n_cluster))
        _initial_distances(
            np.asarray(X),
            cluster_sizes,
            self.alpha,
            out=self.distances[:n_cluster, :n_cluster],
        )

        np.fill_diagonal(self.distances, 0)

        # set up left and right neighbors
        # special case for clusters 0 and n_cluster-1 to allow for cyclic merging
        self.left = np.zeros(2 * n_cluster - 1, dtype=int)
        self.left[:n_cluster] = np.arange(-1, n_cluster - 1)
        self.left[0] = n_cluster - 1

        self.right = np.zeros(2 * n_cluster - 1, dtype=int)
        self.right[:n_cluster] = np.arange(1, n_cluster + 1)
        self.right[n_cluster - 1] = 0

        # True means that a cluster has not been merged_
        self.open = np.ones(2 * n_cluster - 1, dtype=bool)

        # which clusters were merged_ at each step
        self.merged_ = np.empty((n_cluster - 1, 2))

        # set initial gof_ value
        cluster_idx = np.arange(n_cluster)
        self.gof_ = np.array(
            [
                np.sum(
                    self.distances[cluster_idx, self.left[:n_cluster]]
                    + self.distances[cluster_idx, self.right[:n_cluster]]
                )
            ]
        )

        # change point progression
        self.progression = np.empty((n_cluster, n_cluster + 1))
        self.progression[0, 0] = 0
        # N + 1 for cyclic mergers
        np.cumsum(self.sizes[:n_cluster], out=self.progression[0, 1:])

        # array to specify the starting point of a cluster
        self.lm = np.zeros(2 * n_cluster - 1, dtype=int)
        self.lm[:n_cluster] = cluster_idx

    def _gof_update(self, i):
        """Compute the updated goodness-of-fit statistic, left cluster given by i.

        ``i`` can be an int, or an np.ndarray of int, in which case the
        statistics for all clusters in ``i`` are returned, as an np.ndarray.
        """
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

        # see how the gof_ value changes if merged_, for all clusters at once
        candidates = np.flatnonzero(self.open[: K + 1])
        if len(candidates) > 0:
            gof_ = self._gof_update(candidates)
            # argmax returns the first maximum, as the loop over clusters did
            best = np.argmax(gof_)
            if gof_[best] > best_fit:
                best_fit = gof_[best]
                i = candidates[best]
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

        # update distances, for all clusters that have not been merged_ at once
        k = np.flatnonzero(self.open[: K + 1])
        n3 = self.sizes[k]
        n = n1 + n2 + n3
        val = (
            (n - n2) * self.distances[i, k]
            + (n - n1) * self.distances[j, k]
            - n3 * self.distances[i, j]
        ) / n
        self.distances[K + 1, k] = val
        self.distances[k, K + 1] = val

    def _get_penalty_func(self) -> Callable:
        """Define penalty function given (possibly string) input."""
        PENALTIES = {"len_penalty": len_penalty, "mean_diff_penalty": mean_diff_penalty}

        if callable(self.penalty):
            return self.penalty

        elif isinstance(self.penalty, str):
            if self.penalty in PENALTIES:
                return PENALTIES[self.penalty]

        raise ValueError(
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


def _initial_distances(
    X: np.ndarray,
    cluster_sizes: np.ndarray,
    alpha: float,
    out: np.ndarray,
    max_block: int = 2**18,
):
    """Calculate the between-within distances of the initial clusters.

    Computes, for all pairs of initial clusters, twice the mean distance between
    the clusters, minus the mean distance within either cluster, where distances
    are the alpha-th power of the euclidean distance, see equation (4) in [1]_.

    Parameters
    ----------
    X : 2D np.ndarray
        data, rows are observations. Observations of a cluster are assumed to be
        contiguous, which holds as cluster membership is required to be sorted.
    cluster_sizes : 1D np.ndarray of int
        number of observations in each cluster, in order of appearance in ``X``
    alpha : float
        moment of the divergence measure, see ``alpha`` parameter of ``EAgglo``
    out : 2D np.ndarray of shape (len(cluster_sizes), len(cluster_sizes))
        array to write the result to, updated by side effect. Entry [i, j] of
        the result is the between-within distance of clusters i and j.
    max_block : int, optional (default=2**18)
        upper bound on the number of pairwise distances computed at once.
        Distances are computed in blocks of observations, so that the full
        matrix of pairwise distances between observations is not materialized,
        which would be quadratic in the number of observations.
    """
    n_cluster = len(cluster_sizes)

    # observations where each cluster starts
    starts = np.zeros(n_cluster, dtype=int)
    np.cumsum(cluster_sizes[:-1], out=starts[1:])

    # if every observation is its own cluster, the sum of distances between two
    # clusters is the distance between the observations, and summing the blocks
    # of the distance matrix that the clusters span is not needed
    singletons = cluster_sizes.max() == 1

    # number of observations per block, at least one cluster per block
    block_size = max(1, max_block // max(len(X), 1))

    first = 0
    while first < n_cluster:
        # take as many clusters as fit into a block of observations
        last = first + 1
        n_obs = cluster_sizes[first]
        while last < n_cluster and n_obs + cluster_sizes[last] <= block_size:
            n_obs += cluster_sizes[last]
            last += 1

        start = starts[first]
        block = cdist(X[start : start + n_obs], X, "euclidean")
        if alpha != 1:
            block = np.power(block, alpha)

        if singletons:
            out[first:last] = block
        else:
            # sum over the observations of each cluster, in both dimensions
            block = np.add.reduceat(block, starts, axis=1)
            out[first:last] = np.add.reduceat(block, starts[first:last] - start, axis=0)

        first = last

    if singletons:
        # the within-cluster distances are zero, as the clusters are singletons
        out *= 2
        return

    # mean distance between the observations of each pair of clusters
    out /= np.outer(cluster_sizes, cluster_sizes)

    # mean distance within each cluster, the diagonal of the above.
    # copied, since the diagonal is a view, and out is updated in place
    within = np.diag(out).copy()

    out *= 2
    out -= within[:, None]
    out -= within[None, :]


def len_penalty(x: pd.DataFrame) -> int:
    """Penalize goodness-of-fit statistic for number of change points."""
    return -len(x)


def mean_diff_penalty(x: pd.DataFrame) -> float:
    """Penalize goodness-of-fit statistic.

    Favors segmentations with larger sizes, while taking into consideration the size of
    the new segments.
    """
    return np.mean(np.diff(np.sort(x)))
