# -*- coding: utf-8 -*-
"""E-agglo."""

import numpy as np
import pandas as pd

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
        N_ = len(u)  # number of clusters

        for i in range(
            N_
        ):  # relabel clusters to be consecutive numbers (when user specified)
            member_[np.where(member_ == u[i])[0]] = i

        # check if sorted.

        sizes_ = np.repeat(0, 2 * N_)
        sizes_[:N_] = [
            sum(member_ == i) for i in range(N_)
        ]  # calculate initial cluster sizes_

        # array of within distances
        within = [
            get_within(
                X.loc[
                    member_ == i,
                ],
                self.alpha,
            )
            for i in range(N_)
        ]

        # dataframe of between between-within distances
        D_ = pd.DataFrame(index=range(2 * N_), columns=range(2 * N_))

        for i in range(N_):
            for j in range(N_):
                between = get_between(
                    X.loc[
                        member_ == i,
                    ],
                    X.loc[
                        member_ == j,
                    ],
                    self.alpha,
                )
                D_.iloc[i, j] = D_.iloc[j, i] = 2 * between - within[i] - within[j]

        # set up left and right neighbors
        # special case for clusters 0 and N_-1 to allow for cyclic merging
        left_ = np.repeat(0, 2 * N_ - 1)
        left_[:N_] = [i - 1 if i - 1 >= 0 else N_ - 1 for i in range(N_)]
        right_ = np.repeat(0, 2 * N_ - 1)
        right_[:N_] = [i + 1 if i + 1 < N_ else 0 for i in range(N_)]

        # True means that a cluster has not been merged
        open_ = np.array([True for _ in range(2 * N_ - 1)])

        # which clusters were merged at each step
        merged_ = pd.DataFrame(index=range(N_ - 1), columns=range(2))

        # set initial GOF value
        fit_ = sum([D_.iloc[i, left_[i]] + D_.iloc[i, right_[i]] for i in range(N_)])

        # change point progression
        progression_ = pd.DataFrame(index=range(N_), columns=range(N_))
        progression_.iloc[0, ] = [
            sum(sizes_[:i]) if i > 0 else 0 for i in range(N_)
        ]  # FIXME: does this need to be N_+1, just N?

        # array to specify the starting point of a cluster
        lm_ = np.repeat(0, 2 * N_ - 1)
        lm_[:N_] = range(N_)

        # store to self
        self.member_ = member_
        self.N_ = N_
        self.sizes_ = sizes_
        self.D_ = D_
        self.left_ = left_
        self.right_ = right_
        self.open_ = open_
        self.merged_ = merged_
        self.fit_ = fit_
        self.progression_ = progression_
        self.lm_ = lm_


def get_within(X, alpha):
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


def get_between(X, Y, alpha):
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
