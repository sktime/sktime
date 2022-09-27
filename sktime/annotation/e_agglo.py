# -*- coding: utf-8 -*-
"""E-Agglo: agglomerative clustering algorithm that preserves observation order."""

import warnings

import numpy as np

from sktime.transformations.base import BaseTransformer

__author__ = ["KatieBuc"]
__all__ = ["EAGGLO"]


class EAGGLO(BaseTransformer):
    """
    Hierarchical agglomerative estimation of multiple change points, outlined in [1]_.

    E-Agglo is a non-parametric clustering approach for multivariate timeseries, where
    neighboring segments are sequentially merged to maximize a goodness-of-fit
    statistic. Unlike most general purpose agglomerative clustering algorithms, this
    procedure preserves the time ordering of the observations.

    This method can detect any distributional change within an independent sequence,
    and does not make any distributional assumptions (beyond the existence of an
    alpha-th moment). Estimation is performed in a manner that simultaneously
    identifies both the number and locations of change points.

    Parameters
    ----------
    member : array_like (default=None)
        1D `array_like` representing the initial cluster membership for input
        data, X.
    alpha : float (default=1.0)
        fixed constant alpha in (0, 2) used in the divergence measure, as the
        alpha-th absolute moment, see equation (4) in [1]_.
    penalty : functional (default=None)
        function that defines a penalization of  the sequence of goodness-of-fit
        statistics, when overfitting is a concern.

    Attributes
    ----------
    merged : array_like
        2D `array_like` outlining which clusters were merged at each step.
    gof : float
        goodness-of-fit statistic for current clsutering.
    cluster : array_like
        1D `array_like` specifying which cluster each row of input data
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
    >>> from sktime.annotation.e_agglo import EAGGLO
    >>> model = EAGGLO()
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
        super(EAGGLO, self).__init__()

    def _process_data(self, X):
        """Initialize parameters and store to self."""
        _member = (
            self.member if self.member is not None else np.array(range(X.shape[0]))
        )

        u = np.sort(np.unique(_member))  # unique array of cluster labels
        n_cluster = len(u)  # number of clusters

        for i in range(
            n_cluster
        ):  # relabel clusters to be consecutive numbers (when user specified)
            _member[np.where(_member == u[i])[0]] = i

        # check if sorted.

        sizes = np.repeat(0, 2 * n_cluster)
        sizes[:n_cluster] = [
            sum(_member == i) for i in range(n_cluster)
        ]  # calculate initial cluster sizes

        # array of within distances
        within = [
            get_within_distance(
                X.loc[
                    _member == i,
                ],
                self.alpha,
            )
            for i in range(n_cluster)
        ]

        # dataframe of between between-within distances
        distances = np.empty((2 * n_cluster, 2 * n_cluster))

        for i in range(n_cluster):
            for j in range(n_cluster):
                between = get_between_distance(
                    X.loc[
                        _member == i,
                    ],
                    X.loc[
                        _member == j,
                    ],
                    self.alpha,
                )
                distances[i, j] = distances[j, i] = 2 * between - within[i] - within[j]

        np.fill_diagonal(distances, 0)

        # set up left and right neighbors
        # special case for clusters 0 and n_cluster-1 to allow for cyclic merging
        left = np.repeat(0, 2 * n_cluster - 1)
        left[:n_cluster] = [
            i - 1 if i - 1 >= 0 else n_cluster - 1 for i in range(n_cluster)
        ]
        right = np.repeat(0, 2 * n_cluster - 1)
        right[:n_cluster] = [
            i + 1 if i + 1 < n_cluster else 0 for i in range(n_cluster)
        ]

        # True means that a cluster has not been merged
        open = np.array([True for _ in range(2 * n_cluster - 1)])

        # which clusters were merged at each step
        merged = np.empty((n_cluster - 1, 2))

        # set initial GOF value
        gof = np.array(
            [
                sum(
                    [
                        distances[i, left[i]] + distances[i, right[i]]
                        for i in range(n_cluster)
                    ]
                )
            ]
        )

        # change point progression
        progression = np.empty((n_cluster, n_cluster + 1))
        progression[0, :] = [
            sum(sizes[:i]) if i > 0 else 0 for i in range(n_cluster + 1)
        ]  # N + 1 for cyclic mergers

        # array to specify the starting point of a cluster
        lm = np.repeat(0, 2 * n_cluster - 1)
        lm[:n_cluster] = range(n_cluster)

        # store to self
        self._member = _member
        self.n_cluster = n_cluster
        self.sizes = sizes
        self.distances = distances
        self.left = left
        self.right = right
        self.open = open
        self.merged = merged
        self.gof = gof
        self.progression = progression
        self.lm = lm

    def _gof_update(self, i: int):
        """Compute the updated goodness-of-fit statistic, left cluster given by i."""
        fit = self.gof[-1]
        j = self.right[i]

        # get new left and right clusters
        rr = self.right[j]
        ll = self.left[i]

        # remove unneeded values in the GOF
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

    def _find_closest(self, K: int):
        """Determine which clusters will be merged, for K clusters.

        Greedily optimize the goodness-of-fit statistic by merging the pair of adjacent
        clusters that results in the largest increase of the statistic's value.

        Returns
        -------
        result
            tuple of left cluster and right cluster index values
        """
        best_fit = -1e10
        result = (0, 0)

        # iterate through each cluster to see how the GOF value changes if merged
        for i in range(K + 1):
            if self.open[i]:
                gof = self._gof_update(i)
                if gof > best_fit:
                    best_fit = gof
                    result = (i, self.right[i])

        self.gof = np.append(self.gof, best_fit)
        return result

    def _update_distances(self, i: int, j: int, K: int):
        """Update distance from new cluster to other clusters, store to self."""
        # which clusters were merged, info only
        self.merged[K - self.n_cluster + 1, 0] = (
            -i if i <= self.n_cluster else i - self.n_cluster
        )
        self.merged[K - self.n_cluster + 1, 1] = (
            -j if j <= self.n_cluster else j - self.n_cluster
        )

        # update left and right neighbors
        ll = self.left[i]
        rr = self.right[j]
        self.left[K + 1] = ll
        self.right[K + 1] = rr
        self.right[ll] = K + 1
        self.left[rr] = K + 1

        # update information about which clusters have been merged
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

    def _fit(self, X, y=None):
        """Find optimally clustered segments.

        First, by determining which pairs of adjacent clusters will be merged. Then,
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

        assert self.alpha > 0 and self.alpha <= 2

        self._process_data(X)

        # find which clusters optimize the GOF and then update the distances
        for K in range(self.n_cluster - 1, 2 * self.n_cluster - 2):
            i, j = self._find_closest(K)
            self._update_distances(i, j, K)

        # penalize the GOF statistic
        if self.penalty is not None:
            penalty_func = get_penalty_func(self.penalty)
            cps = [
                list(
                    filter(
                        lambda v: v == v,
                        self.progression[
                            i,
                        ],
                    )
                )
                for i in range(len(self.progression))
            ]
            self.gof += list(map(penalty_func, cps))

        # get the set of change points for the "best" clustering
        idx = np.argmax(self.gof)
        self._estimates = np.sort(
            list(
                filter(
                    lambda v: v == v,
                    self.progression[
                        idx,
                    ],
                )
            )
        )

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
            self.cluster = get_cluster(self._estimates)
        else:
            tmp = get_cluster(np.append([0], self._estimates))
            self.cluster = np.append(tmp, np.repeat(0, X.shape[0] - len(tmp)))

        return self

    def _transform(self, X, y=None):
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
            new_eagglo = EAGGLO(
                member=self.member,
                alpha=self.alpha,
                penalty=self.penalty,
            ).fit(X_full)
            warnings.warn(
                "Warning: Input data X differs from that given to fit(). "
                "Refitting with both the data in fit and new input data, not storing "
                "updated public class attributes. For this, explicitly use fit(X) or "
                "fit_transform(X)."
            )
            return new_eagglo.cluster

        return self.cluster


def get_within_distance(X, alpha):
    """Calculate within cluster distance."""
    n = X.shape[0]
    return sum(
        np.power(
            np.sqrt(
                sum(
                    (
                        X.iloc[
                            i,
                        ]
                        - X.iloc[
                            j,
                        ]
                    )
                    * (
                        X.iloc[
                            i,
                        ]
                        - X.iloc[
                            j,
                        ]
                    )
                )
            ),
            alpha,
        )
        for j in range(n)
        for i in range(n)
    ) / (n * n)


def get_between_distance(X, Y, alpha):
    """Calculate between cluster distance."""
    n = X.shape[0]
    m = Y.shape[0]
    return sum(
        np.power(
            np.sqrt(
                sum(
                    (
                        X.iloc[
                            i,
                        ]
                        - Y.iloc[
                            j,
                        ]
                    )
                    * (
                        X.iloc[
                            i,
                        ]
                        - Y.iloc[
                            j,
                        ]
                    )
                )
            ),
            alpha,
        )
        for j in range(m)
        for i in range(n)
    ) / (m * n)


def penalty1(x):
    """Penalize goodness-of-fit statistic for number of change points."""
    return -len(x)


def penalty2(x):
    """Penalize goodness-of-fit statistic.

    Favors segmentations with larger sizes, while taking into consideration
    the size of the new segments.
    """
    return np.mean(np.diff(np.sort(x)))


def get_penalty_func(penalty):
    """Define penalty function given (possibly string) input."""
    PENALTIES = {"penalty1": penalty1, "penalty2": penalty2}

    if callable(penalty):
        return penalty

    elif isinstance(penalty, str):
        if penalty in PENALTIES.keys():
            return PENALTIES[penalty]

    raise Exception(
        "penalty must be callable or string values 'penalty1' or 'penalty2'."
    )
