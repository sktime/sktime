# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Abstract base class for unsupervised sequence aligners.

This covers both pairwise and multiple sequence aligners.

    class name: BaseAligner

Scitype defining methods:
    fitting              - fit(self, X, Z=None)
    get alignment (iloc) - get_alignment()
    get alignment (loc)  - get_alignment_loc()
    get aligned series   - get_aligned()
    get distance (float) - get_distance()
    get distance matrix  - get_distance_matrix()

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()
"""

__author__ = ["fkiraly"]


from sktime.alignment.utils.utils_align import convert_align_to_align_loc, reindex_iloc
from sktime.base import BaseEstimator
from sktime.datatypes import check_is_scitype, convert
from sktime.datatypes._dtypekind import DtypeKind


class BaseAligner(BaseEstimator):
    """Base class for all unsupervised sequence aligners."""

    _tags = {
        "object_type": "aligner",  # type of object
        "capability:multiple-alignment": False,  # can align more than two sequences?
        "capability:distance": False,  # does compute/return overall distance?
        "capability:distance-matrix": False,  # does compute/return distance matrix?
        "capability:unequal_length": True,  # can align sequences of unequal length?
        "alignment_type": "full",  # does the aligner produce full or partial alignment
        "X_inner_mtype": "df-list",  # mtype of X expected by _fit
    }

    def __init__(self):
        """Construct the class."""
        self._is_fitted = False
        self._X = None

        super().__init__()

    def fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Stores `X` and `Z` to self._X and self._Z, respectively.
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : list of pd.DataFrame (Series) of length n
            collection of series to align
        Z : pd.DataFrame with n rows, optional
            metadata, i-th row of Z corresponds to i-th element of X
        """
        # if fit is called, estimator is reset, including fitted state
        self.reset()

        METADATA_TO_QUERY = ["is_equal_length", "n_instances", "feature_kind"]
        valid, msg, X_metadata = check_is_scitype(
            X, scitype="Panel", return_metadata=METADATA_TO_QUERY, var_name="X"
        )

        if not valid:
            raise TypeError(msg)

        if DtypeKind.CATEGORICAL in X_metadata["feature_kind"]:
            raise TypeError("Aligners do not support categorical features in X.")

        self._check_capabilities(X_metadata)

        X_mtype = X_metadata["mtype"]
        X_inner_mtype = self.get_tag("X_inner_mtype")

        self._X_mtype = X_mtype

        X_inner = convert(
            X, from_type=X_mtype, to_type=X_inner_mtype, as_scitype="Panel"
        )

        self._fit(X=X_inner, Z=Z)

        # convert X to df-list for use in get_aligned, get_alignment_loc
        if X_inner_mtype != "df-list":
            self._X = convert(
                X, from_type=X_mtype, to_type="df-list", as_scitype="Panel"
            )
        else:
            self._X = X_inner

        self._Z = Z

        self._is_fitted = True

        return self

    def _fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

            core logic

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : list of pd.DataFrame (Series) of length n
            collection of series to align
        Z : pd.DataFrame with n rows, optional
            metadata, i-th row of Z corresponds to i-th element of X
        """
        raise NotImplementedError

    def get_alignment(self):
        """Return alignment for sequences/series passed in fit (iloc indices).

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain iloc index of X[i] mapped to alignment coordinate for alignment
        """
        self.check_is_fitted()
        return self._get_alignment()

    def _get_alignment(self):
        """Return alignment for sequences/series passed in fit (iloc indices).

            core logic

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Accesses in self:
            Fitted model attributes ending in "_".

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

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain loc index of X[i] mapped to alignment coordinate for alignment
        """
        self.check_is_fitted()
        return self._get_alignment_loc()

    def _get_alignment_loc(self):
        """Return alignment for sequences/series passed in fit (loc indices).

            core logic

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Accesses in self:
            Fitted model attributes ending in "_".

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain loc index of X[i] mapped to alignment coordinate for alignment
        """
        if not hasattr(self, "_X"):
            # defensive error - fit should store X to self._X
            raise RuntimeError(
                "fit needs to store X to self._X when using default get_alignment_loc"
            )

        X = self._X

        align = self.get_alignment()

        return convert_align_to_align_loc(align, X)

    def get_aligned(self):
        """Return aligned version of sequences passed to fit.

        Behaviour: returns aligned version of unaligned sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Returns
        -------
        X_aligned_list: list of pd.DataFrame in sequence format
            of length n, indices corresponding to indices of X passed to fit
            i-th element is re-indexed, aligned version of X[i]
        """
        self.check_is_fitted()
        return self._get_aligned()

    def _get_aligned(self):
        """Return aligned version of sequences passed to fit.

            core logic

        Behaviour: returns aligned version of unaligned sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Accesses in self:
            Fitted model attributes ending in "_".

        Returns
        -------
        X_aligned_list: list of pd.DataFrame in sequence format
            of length n, indices corresponding to indices of X passed to fit
            i-th element is re-indexed, aligned version of X[i]
        """
        if not hasattr(self, "_X"):
            # defensive error - fit should store X to self._X
            raise RuntimeError(
                "fit needs to store X to self._X when using default get_aligned"
            )

        X = self._X
        align = self.get_alignment()

        X_aligned_list = []

        for i, Xi in enumerate(X):
            indi = "ind" + str(i)
            X_aligned_list += [reindex_iloc(Xi, align[indi], copy=True)]

        return X_aligned_list

    def get_distance(self):
        """Return overall distance of alignment.

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        self.check_is_fitted()
        return self._get_distance()

    def _get_distance(self):
        """Return overall distance of alignment.

            core logic

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)
        Accesses in self:
            Fitted model attributes ending in "_".

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        raise NotImplementedError

    def get_distance_matrix(self):
        """Return distance matrix of alignment.

        Behaviour: returns pairwise distance matrix of alignment distances
            not all aligners will return or implement this (optional)

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self._is_fitted

        Returns
        -------
        distmat: an (n x n) np.array of floats, where n is length of X passed to fit
            [i,j]-th entry is alignment distance between X[i] and X[j] passed to fit
        """
        self.check_is_fitted()
        return self._get_distance_matrix()

    def _get_distance_matrix(self):
        """Return distance matrix of alignment.

            core logic

        Behaviour: returns pairwise distance matrix of alignment distances
            not all aligners will return or implement this (optional)

        Accesses in self:
            Fitted model attributes ending in "_".

        Returns
        -------
        distmat: an (n x n) np.array of floats, where n is length of X passed to fit
            [i,j]-th entry is alignment distance between X[i] and X[j] passed to fit
        """
        # the default implementation assumes
        # that the aligner can only align two sequences
        if self.get_tag("capability:multiple-alignment", False):
            raise NotImplementedError

        import numpy as np

        dist = self.get_distance()

        distmat = np.zeros((2, 2), dtype="float")
        distmat[0, 1] = dist
        distmat[1, 0] = dist

        return distmat

    def _check_capabilities(self, X_metadata):
        """Check if the aligner can align the input sequences.

        Parameters
        ----------
        X_metadata : dict
            metadata of the input sequences
        """
        # if aligner does not support unequal length sequences
        # and X has unequal length, raise error
        X_equal_length = X_metadata["is_equal_length"]
        if not self.get_tag("capability:unequal_length", True) and not X_equal_length:
            raise ValueError(
                f"Aligner {self.__class__.__name__} instance does not support "
                "alignment of unequal length sequences, but X passed "
                "had unequal length. "
                "Presence or lack of this capability may be depend on "
                "hyper-parameters, especially for composites. Please consult the "
                "documentation of the aligner for more information."
            )

        # if aligner does not support multiple alignment
        # and X has more than two sequences, raise error
        n_instances = X_metadata["n_instances"]
        if not self.get_tag("capability:multiple-alignment", False) and n_instances > 2:
            raise ValueError(
                f"Aligner {self.__class__.__name__} instance does not support "
                "alignment of multiple sequences, but X passed "
                f"had {n_instances} sequences. "
                "Presence or lack of this capability may be depend on "
                "hyper-parameters, especially for composites. Please consult the "
                "documentation of the aligner for more information."
            )
