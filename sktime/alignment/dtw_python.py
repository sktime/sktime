"""Interface module to dtw-python package.

Exposes basic interface, excluding multivariate case.
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.alignment.base import BaseAligner


class AlignerDTW(BaseAligner):
    """Aligner interface for dtw-python.

    Behaviour: computes the full alignment between X[0] and X[1]
        assumes pairwise alignment (only two series) and univariate
        if multivariate series are passed:
        alignment is computed on univariate series with variable_to_align;
        if this is not set, defaults to the first variable of X[0]
        raises an error if variable_to_align is not present in X[0] or X[1]

    Parameters
    ----------
    dist_method : str, optional, default = "euclidean"
        distance function to use, a distance on real n-space
        one of the functions in ``scipy.spatial.distance.cdist``
    step_pattern : str, optional, or dtw_python stepPattern object, optional
        step pattern to use in time warping
        one of: 'symmetric1', 'symmetric2' (default), 'asymmetric',
        and dozens of other more non-standard step patterns;
        list can be displayed by calling help(stepPattern) in dtw
    window_type : string, the chosen windowing function
        "none", "itakura", "sakoechiba", or "slantedband"
        "none" (default) - no windowing
        "sakoechiba" - a band around main diagonal
        "slantedband" - a band around slanted diagonal
        "itakura" - Itakura parallelogram
    window_size: int, optional, default=None
        size of the window if a windowing function is used
        if None and window_type="sakoechiba", defaults to 10% of series length
    open_begin : boolean, optional, default=False
    open_end: boolean, optional, default=False
        whether to perform open-ended alignments
        open_begin = whether alignment open ended at start (low index)
        open_end = whether alignment open ended at end (high index)
    variable_to_align : string, default = first variable in X[0] as passed to fit
        which variable to use for univariate alignment

    Examples
    --------
    Basic usage example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.alignment.dtw_python import AlignerDTW
    >>> X = [
    ...     pd.DataFrame({'col1': np.random.randn(100)}),
    ...     pd.DataFrame({'col1': np.random.randn(100)})
    ... ]
    >>> aligner = AlignerDTW(dist_method='euclidean', step_pattern='symmetric2')
    >>> aligner.fit(X)
    AlignerDTW(...)
    >>> alignment_df = aligner.get_alignment()

    Advanced usage example with open-ended alignment:
    >>> aligner_advanced = AlignerDTW(
    ...     dist_method='cityblock',
    ...     window_type='sakoechiba',
    ...     window_size=10,
    ...     step_pattern='asymmetric',
    ...     open_begin=True,
    ...     open_end=True,
    ... )
    >>> X_advanced = [
    ...     pd.DataFrame({'col1': np.random.randn(150)}),
    ...     pd.DataFrame({'col1': np.random.randn(150)})
    ... ]
    >>> aligner_advanced.fit(X_advanced)
    AlignerDTW(...)
    >>> alignment_df_advanced = aligner_advanced.get_alignment()
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["tonigi", "fkiraly"],  # tonigi for dtw-python
        "python_dependencies": "dtw-python",
        # estimator type
        # --------------
        "capability:multiple-alignment": False,  # can align more than two sequences?
        "capability:distance": True,  # does compute/return overall distance?
        "capability:distance-matrix": True,  # does compute/return distance matrix?
        "capability:unequal_length": True,  # can align sequences of unequal length?
        "alignment_type": "partial",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        dist_method="euclidean",
        step_pattern="symmetric2",
        window_type="none",
        window_size=None,
        open_begin=False,
        open_end=False,
        variable_to_align=None,
    ):
        """Construct instance."""
        # added manually since dtw-python has an import alias
        # default check from super.__init__ does not allow aliases
        super().__init__()

        self.dist_method = dist_method
        self.step_pattern = step_pattern
        self.window_type = window_type
        self.window_size = window_size
        self.open_begin = open_begin
        self.open_end = open_end
        self.variable_to_align = variable_to_align

        if open_end or open_begin:
            self.set_tags(**{"alignment_type": "partial"})
        else:
            self.set_tags(**{"alignment_type": "full"})

    def _fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

            core logic

        Writes to self:
            alignment : computed alignment from dtw package (nested struct)

        Parameters
        ----------
        X: list of pd.DataFrame (sequence) of length n - panel of series to align
        Z: pd.DataFrame with n rows, optional; metadata, row correspond to indices of X
        """
        # soft dependency import of dtw
        from dtw import dtw

        # these variables from self are accessed
        dist_method = self.dist_method
        step_pattern = self.step_pattern
        window_type = self.window_type
        window_size = self.window_size
        open_begin = self.open_begin
        open_end = self.open_end
        var_to_align = self.variable_to_align

        # If sakoi-cheba window is used, window_size must be set or given a default
        # value equal to 10 % of the length of the series
        if window_type == "sakoechiba":
            if window_size is None:
                window_size = int(0.1 * len(X[0]))

        # shorthands for 1st and 2nd series
        XA = X[0]
        XB = X[1]

        # retrieve column to align from data frames, convert to np.array
        if var_to_align is None:
            var_to_align = XA.columns.values[0]

        if var_to_align not in XA.columns.values:
            raise ValueError(
                f"X[0] does not have variable {var_to_align} used for alignment"
            )
        if var_to_align not in XB.columns.values:
            raise ValueError(
                f"X[1] does not have variable {var_to_align} used for alignment"
            )

        XA_np = XA[var_to_align].values
        XB_np = XB[var_to_align].values

        # pass to the interfaced dtw function and store to self
        alignment = dtw(
            XA_np,
            XB_np,
            dist_method=dist_method,
            step_pattern=step_pattern,
            window_type=window_type,
            window_args={"window_size": window_size},
            open_begin=open_begin,
            open_end=open_end,
            keep_internals=True,
        )

        self.alignment_ = alignment
        self.variable_to_align_ = var_to_align  # changed only if was None

        return self

    def _get_alignment(self):
        """Return alignment for sequences/series passed in fit (iloc indices).

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain iloc index of X[i] mapped to alignment coordinate for alignment
        """
        # retrieve alignment
        alignment = self.alignment_

        # convert to required data frame format and return
        aligndf = pd.DataFrame({"ind0": alignment.index1, "ind1": alignment.index2})

        return aligndf

    def _get_distance(self):
        """Return overall distance of alignment.

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        return self.alignment_.distance

    def _get_distance_matrix(self):
        """Return distance matrix of alignment.

        Behaviour: returns pairwise distance matrix of alignment distances
            not all aligners will return or implement this (optional)

        Returns
        -------
        distmat: a (2 x 2) np.array of floats
            [i,j]-th entry is alignment distance between X[i] and X[j] passed to fit
        """
        # since dtw does only pairwise alignments, this is always a 2x2 matrix
        distmat = np.zeros((2, 2), dtype="float")
        distmat[0, 1] = self.alignment_.distance
        distmat[1, 0] = self.alignment_.distance

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for AlignerDTWdist."""
        params1 = {}
        params2 = {"step_pattern": "symmetric1"}
        params3 = {"window_type": "sakoechiba"}

        return [params1, params2, params3]


class AlignerDTWfromDist(BaseAligner):
    """Aligner interface for dtw-python using pairwise transformer.

    Uses transformer for computation of distance matrix passed to alignment.

    Parameters
    ----------
    dist_trafo: estimator following the pairwise transformer template
        i.e., instance of concrete class implementing template BasePairwiseTransformer
    step_pattern : str, optional, default = "symmetric2",
        or dtw_python stepPattern object, optional
        step pattern to use in time warping
        one of: 'symmetric1', 'symmetric2' (default), 'asymmetric',
        and dozens of other more non-standard step patterns;
        list can be displayed by calling help(stepPattern) in dtw
    window_type: str  optional, default = "none"
        the chosen windowing function
        "none", "itakura", "sakoechiba", or "slantedband"
        "none" (default) - no windowing
        "sakoechiba" - a band around main diagonal
        "slantedband" - a band around slanted diagonal
        "itakura" - Itakura parallelogram
    window_size: int, optional, default=None
        size of the window if a windowing function is used
        if None and window_type="sakoechiba", defaults to 10% of series length
    open_begin : boolean, optional, default=False
    open_end: boolean, optional, default=False
        whether to perform open-ended alignments
        open_begin = whether alignment open ended at start (low index)
        open_end = whether alignment open ended at end (high index)

    Examples
    --------
    Basic usage example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.alignment.dtw_python import AlignerDTWfromDist
    >>> from sktime.dists_kernels import ScipyDist
    >>> X = [
    ...     pd.DataFrame({'col1': np.random.randn(100)}),
    ...     pd.DataFrame({'col1': np.random.randn(100)})
    ... ]
    >>> dist_trafo = ScipyDist()
    >>> aligner = AlignerDTWfromDist(dist_trafo=dist_trafo, step_pattern='symmetric2')
    >>> aligner.fit(X)
    AlignerDTWfromDist(...)
    >>> alignment_df = aligner.get_alignment()

    Advanced usage example with custom distance transformation:
    >>> dist_trafo_custom = ScipyDist('cityblock')
    >>> aligner_custom = AlignerDTWfromDist(
    ...     dist_trafo=dist_trafo_custom,
    ...     window_type='sakoechiba',
    ...     window_size=10,
    ... )
    >>> X_custom = [
    ...     pd.DataFrame({'col1': np.random.randn(200)}),
    ...     pd.DataFrame({'col1': np.random.randn(200)})
    ... ]
    >>> aligner_custom.fit(X_custom)
    AlignerDTWfromDist(...)
    >>> alignment_df_custom = aligner_custom.get_alignment()
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["tonigi", "fkiraly"],  # tonigi for dtw-python
        "python_dependencies": "dtw-python",
        # estimator type
        # --------------
        "capability:multiple-alignment": False,  # can align more than two sequences?
        "capability:distance": True,  # does compute/return overall distance?
        "capability:distance-matrix": True,  # does compute/return distance matrix?
    }

    def __init__(
        self,
        dist_trafo,
        step_pattern="symmetric2",
        window_type="none",
        window_size=None,
        open_begin=False,
        open_end=False,
    ):
        super().__init__()

        self.dist_trafo = dist_trafo
        self.dist_trafo_ = self.dist_trafo.clone()
        self.step_pattern = step_pattern
        self.window_type = window_type
        self.window_size = window_size
        self.open_begin = open_begin
        self.open_end = open_end

    def _fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

            core logic

        Writes to self:
            alignment : computed alignment from dtw package (nested struct)

        Parameters
        ----------
        X: list of pd.DataFrame (sequence) of length n - panel of series to align
        Z: pd.DataFrame with n rows, optional; metadata, row correspond to indices of X
        """
        # soft dependency import of dtw
        from dtw import dtw

        # these variables from self are accessed
        dist_trafo = self.dist_trafo_
        step_pattern = self.step_pattern
        window_type = self.window_type
        window_size = self.window_size
        open_begin = self.open_begin
        open_end = self.open_end

        # If sakoi-cheba window is used, window_size must be set or given a default
        # value equal to 10 % of the length of the series
        if window_type == "sakoechiba":
            if window_size is None:
                window_size = int(0.1 * len(X[0]))

        # shorthands for 1st and 2nd sequence
        XA = X[0]
        XB = X[1]

        # compute distance matrix using cdist
        distmat = dist_trafo(XA, XB)

        # pass to the interfaced dtw function and store to self
        alignment = dtw(
            distmat,
            step_pattern=step_pattern,
            window_type=window_type,
            window_args={"window_size": window_size},
            open_begin=open_begin,
            open_end=open_end,
            keep_internals=True,
        )

        self.alignment_ = alignment

        return self

    def _get_alignment(self):
        """Return alignment for sequences/series passed in fit (iloc indices).

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain iloc index of X[i] mapped to alignment coordinate for alignment
        """
        # retrieve alignment
        alignment = self.alignment_

        # convert to required data frame format and return
        aligndf = pd.DataFrame({"ind0": alignment.index1, "ind1": alignment.index2})

        return aligndf

    def _get_distance(self):
        """Return overall distance of alignment.

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        return self.alignment_.distance

    def _get_distance_matrix(self):
        """Return distance matrix of alignment.

        Behaviour: returns pairwise distance matrix of alignment distances
            not all aligners will return or implement this (optional)

        Returns
        -------
        distmat: a (2 x 2) np.array of floats
            [i,j]-th entry is alignment distance between X[i] and X[j] passed to fit
        """
        # since dtw does only pairwise alignments, this is always
        distmat = np.zeros((2, 2), dtype="float")
        distmat[0, 1] = self.alignment_.distance
        distmat[1, 0] = self.alignment_.distance

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for AlignerDTWdist."""
        # importing inside to avoid circular dependencies
        from sktime.dists_kernels import ScipyDist

        params1 = {"dist_trafo": ScipyDist()}
        params2 = {"dist_trafo": ScipyDist("cityblock"), "step_pattern": "symmetric1"}
        params3 = {"dist_trafo": ScipyDist(), "window_type": "sakoechiba"}

        return [params1, params2, params3]
