# -*- coding: utf-8 -*-
"""Abstract base class for unsupervised sequence aligners.

This covers both pairwise and multiple sequence aligners.
"""

__author__ = ["fkiraly"]


from copy import deepcopy

from sktime.base import BaseEstimator

from sktime.alignment.utils.utils_align import convert_align_to_align_loc
from sktime.alignment.utils.utils_align import reindex_iloc


class BaseAligner(BaseEstimator):
    """Base class for all unsupervised sequence aligners."""

    _tags = {
        "capability:multiple-alignment": False,  # can align more than two sequences?
        "capability:distance": False,  # does compute/return overall distance?
        "capability:distance-matrix": False,  # does compute/return distance matrix?
    }

    def __init__(self):
        """Construct the class."""
        self._is_fitted = False
        self._X = None

        super(BaseAligner, self).__init__()

    def fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

        Parameters
        ----------
        X: list of pd.DataFrame (sequence) of length n - panel of series to align
        Z: pd.DataFrame with n rows, optional; metadata, row correspond to indices of X

        State change
        ---------------
        creates fitted model (attributes ending in "_")
        sets is_fitted flag to true
        should write X to self if using default implementation of get_alignment_loc
        """
        self._fit(X=X, Z=Z)

        self._is_fitted = True

        return self

    def _fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

            core logic

        Parameters
        ----------
        X: list of pd.DataFrame (sequence) of length n - panel of series to align
        Z: pd.DataFrame with n rows, optional; metadata, row correspond to indices of X

        Writes to self
        --------------
        creates fitted model (attributes ending in "_")
        """
        raise NotImplementedError

    def get_alignment(self):
        """Return alignment for sequences/series passed in fit (iloc indices).

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain iloc index of X[i] mapped to alignment coordinate for alignment
        """
        raise NotImplementedError

    def get_alignment_loc(self):
        """Return alignment for sequences/series passed in fit (loc indices).

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain loc index of X[i] mapped to alignment coordinate for alignment
        """
        if not hasattr(self, "X"):
            raise NotImplementedError(
                "fit needs to store X to self when using default get_aligned"
            )

        X = self.X

        align = self.get_alignment()

        return convert_align_to_align_loc(align, X)

    def get_aligned(self):
        """Return aligned version of sequences passed to fit.

        Behaviour: returns aligned version of unaligned sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Returns
        -------
        X_aligned_list: list of pd.DataFrame in sequence format
            of length n, indices corresponding to indices of X passed to fit
            i-th element is re-indexed, aligned version of X[i]
        """
        if not hasattr(self, "X"):
            raise NotImplementedError(
                "fit needs to store X to self when using default get_aligned"
            )

        X_orig = self.X
        X_aligned_list = deepcopy(X_orig)

        align = self.get_alignment()

        for i, Xi_orig in enumerate(X_aligned_list):

            indi = "ind" + str(i)
            X_aligned_list[i] = self._apply_alignment(Xi_orig, align[indi])

        return X_aligned_list

    def _apply_alignment(self, Xi, align_inds):
        """Apply a column of an alignment to a sequence, private helper function.

        Parameters
        ----------
        Xi: pd.DataFrame, sequence to be realigned/reindexed
        align_inds: pd.Series, iloc indices for realignment/reindexing

        Returns
        -------
        Xi_aligned: pd.DataFrame, Xi aligned/reindexed via align_inds
        """
        Xi_aligned = reindex_iloc(Xi, align_inds)
        Xi_aligned.index = align_inds.index

        return Xi_aligned

    def get_distance(self):
        """Return overall distance of alignment.

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        raise NotImplementedError

    def get_distances(self):
        """Return distance matrix of alignment.

        Behaviour: returns pairwise distance matrix of alignment distances
            not all aligners will return or implement this (optional)

        Returns
        -------
        distmat: an (n x n) np.array of floats, where n is length of X passed to fit
            [i,j]-th entry is alignment distance between X[i] and X[j] passed to fit
        """
        raise NotImplementedError
