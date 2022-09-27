# -*- coding: utf-8 -*-
"""E-agglo."""

import warnings

import numpy as np

from sktime.transformations.base import BaseTransformer

__author__ = ["KatieBuc"]
__all__ = ["EAGGLO"]


class EAGGLO(BaseTransformer):
    """
    Docstring.

    Parameters
    ----------
    ...

    Attributes
    ----------
    ...

    References
    ----------
    .. [1] ...

    Examples
    --------
    ...
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
        """Docstring."""
        member_ = (
            self.member if self.member is not None else np.array(range(X.shape[0]))
        )

        u = np.sort(np.unique(member_))  # unique array of cluster labels
        n_cluster = len(u)  # number of clusters

        for i in range(
            n_cluster
        ):  # relabel clusters to be consecutive numbers (when user specified)
            member_[np.where(member_ == u[i])[0]] = i

        # check if sorted.

        sizes_ = np.repeat(0, 2 * n_cluster)
        sizes_[:n_cluster] = [
            sum(member_ == i) for i in range(n_cluster)
        ]  # calculate initial cluster sizes_

        # array of within distances
        within = [
            get_within_distance(
                X.loc[
                    member_ == i,
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
                        member_ == i,
                    ],
                    X.loc[
                        member_ == j,
                    ],
                    self.alpha,
                )
                distances[i, j] = distances[j, i] = 2 * between - within[i] - within[j]

        np.fill_diagonal(distances, 0)

        # set up left and right neighbors
        # special case for clusters 0 and n_cluster-1 to allow for cyclic merging
        left_ = np.repeat(0, 2 * n_cluster - 1)
        left_[:n_cluster] = [
            i - 1 if i - 1 >= 0 else n_cluster - 1 for i in range(n_cluster)
        ]
        right_ = np.repeat(0, 2 * n_cluster - 1)
        right_[:n_cluster] = [
            i + 1 if i + 1 < n_cluster else 0 for i in range(n_cluster)
        ]

        # True means that a cluster has not been merged
        open_ = np.array([True for _ in range(2 * n_cluster - 1)])

        # which clusters were merged at each step
        merged_ = np.empty((n_cluster - 1, 2))

        # set initial GOF value
        fit_ = np.array(
            [
                sum(
                    [
                        distances[i, left_[i]] + distances[i, right_[i]]
                        for i in range(n_cluster)
                    ]
                )
            ]
        )

        # change point progression
        progression_ = np.empty((n_cluster, n_cluster + 1))
        progression_[0, :] = [
            sum(sizes_[:i]) if i > 0 else 0 for i in range(n_cluster + 1)
        ]  # N + 1 for cyclic mergers

        # array to specify the starting point of a cluster
        lm_ = np.repeat(0, 2 * n_cluster - 1)
        lm_[:n_cluster] = range(n_cluster)

        # store to self
        self.member_ = member_
        self.n_cluster = n_cluster
        self.sizes_ = sizes_
        self.distances = distances
        self.left_ = left_
        self.right_ = right_
        self.open_ = open_
        self.merged_ = merged_
        self.fit_ = fit_
        self.progression_ = progression_
        self.lm_ = lm_

    def _gof_update(self, i):
        """Docstring."""
        fit = self.fit_[-1]
        j = self.right_[i]

        # get new left and right clusters
        rr = self.right_[j]
        ll = self.left_[i]

        # remove unneeded values in the GOF
        fit -= 2 * (
            self.distances[i, j] + self.distances[i, ll] + self.distances[j, rr]
        )

        # get cluster sizes
        n1 = self.sizes_[i]
        n2 = self.sizes_[j]

        # add distance to new left cluster
        n3 = self.sizes_[ll]
        k = (
            (n1 + n3) * self.distances[i, ll]
            + (n2 + n3) * self.distances[j, ll]
            - n3 * self.distances[i, j]
        ) / (n1 + n2 + n3)
        fit += 2 * k

        # add distance to new right
        n3 = self.sizes_[rr]
        k = (
            (n1 + n3) * self.distances[i, rr]
            + (n2 + n3) * self.distances[j, rr]
            - n3 * self.distances[i, j]
        ) / (n1 + n2 + n3)
        fit += 2 * k

        return fit

    def _find_closest(self, K):
        """Docstring."""
        best_fit = -1e10
        result = (0, 0)

        # iterate to see how the GOF value changes
        for i in range(K + 1):
            if self.open_[i]:
                fit_ = self._gof_update(i)
                if fit_ > best_fit:
                    best_fit = fit_
                    result = (i, self.right_[i])

        self.fit_ = np.append(self.fit_, best_fit)
        return result

    def _update_distances(self, i, j, K):
        """Docstring."""
        # which clusters were merged, info only
        self.merged_[K - self.n_cluster + 1, 0] = (
            -i if i <= self.n_cluster else i - self.n_cluster
        )
        self.merged_[K - self.n_cluster + 1, 1] = (
            -j if j <= self.n_cluster else j - self.n_cluster
        )

        # update left and right neighbors
        ll = self.left_[i]
        rr = self.right_[j]
        self.left_[K + 1] = ll
        self.right_[K + 1] = rr
        self.right_[ll] = K + 1
        self.left_[rr] = K + 1

        # update information about which clusters have been merged
        self.open_[i] = False
        self.open_[j] = False

        # assign size to newly created cluster
        n1 = self.sizes_[i]
        n2 = self.sizes_[j]
        self.sizes_[K + 1] = n1 + n2

        # update set of change points
        self.progression_[K - self.n_cluster + 2, :] = self.progression_[
            K - self.n_cluster + 1,
        ]
        self.progression_[K - self.n_cluster + 2, self.lm_[j]] = np.nan
        self.lm_[K + 1] = self.lm_[i]

        # update distances
        for k in range(K + 1):
            if self.open_[k]:
                n3 = self.sizes_[k]
                n = n1 + n2 + n3
                val = (
                    (n - n2) * self.distances[i, k]
                    + (n - n1) * self.distances[j, k]
                    - n3 * self.distances[i, j]
                ) / n
                self.distances[K + 1, k] = val
                self.distances[k, K + 1] = val

    def _fit(self, X, y=None):
        """Find ....

        Parameters
        ----------
        X : np.ArrayLike
            Data for anomaly detection (time series).
        y : pd.Series, optional
            Not used for this unsupervsed method.

        Attributes
        ----------
        fit_
        cluster_
        TODO: change from public to private variables

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
                        self.progression_[
                            i,
                        ],
                    )
                )
                for i in range(len(self.progression_))
            ]
            self.fit_ += list(map(penalty_func, cps))

        # get the set of change points for the "best" clustering
        idx = np.argmax(self.fit_)
        self.estimates_ = np.sort(
            list(
                filter(
                    lambda v: v == v,
                    self.progression_[
                        idx,
                    ],
                )
            )
        )

        # remove change point N+1 if a cyclic merger was performed
        self.estimates_ = (
            self.estimates_[:-1] if self.estimates_[0] != 0 else self.estimates_
        )

        # create final membership vector
        def get_cluster(estimates):
            return np.repeat(
                range(len(np.diff(estimates))), np.diff(estimates).astype(int)
            )

        if self.estimates_[0] == 0:
            self.cluster_ = get_cluster(self.estimates_)
        else:
            tmp = get_cluster(np.append([0], self.estimates_))
            self.cluster_ = np.append(tmp, np.repeat(0, X.shape[0] - len(tmp)))

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
        transformed version of X, representing the original data on a new set of
        coordinates, obtained by multiplying input data by the basis vectors.
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
            return new_eagglo.cluster_

        return self.cluster_


def get_within_distance(X, alpha):
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
    return -len(x)


def penalty2(x):
    return np.mean(np.diff(np.sort(x)))


def get_penalty_func(penalty):
    PENALTIES = {"penalty1": penalty1, "penalty2": penalty2}

    if callable(penalty):
        return penalty

    elif isinstance(penalty, str):
        if penalty in PENALTIES.keys():
            return PENALTIES[penalty]

    raise Exception(
        "penalty must be callable or string values 'penalty1' or 'penalty2'."
    )
