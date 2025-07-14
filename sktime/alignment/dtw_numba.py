"""BaseAligner interface to sktime dtw aligners in distances module."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.alignment.base import BaseAligner


class AlignerDtwNumba(BaseAligner):
    r"""Interface to sktime native dtw aligners, with derivative or weighting.

    Interface to simple dynamic time warping (DTW) alignment,
    and the following weighted/derivative versions:

    * WDTW - weighted dynamic tyme warping - ``weighted=True, derivative=False`
    * DDTW - derivative dynamic time warping - ``weighted=False, derivative=True``
    * WDDTW - weighted derivative dynamic time
      warping - ``weighted=True, derivative=True``

    ``sktime`` interface to the efficient ``numba`` implementations
    provided by ``distance_alignment_path`` in ``sktime.distances``.

    This estimator provides performant implementation of time warping for:
    * time series of equal length
    * the Euclidean pairwise distance

    For unequal length time series, use ``sktime.aligners.AlignerDTW``.
    To use arbitrary pairwise distances, use ``sktime.aligners.AlignerDTWfromDist``.
    (for derivative DTW, pipeline an alignment distance with ``Differencer``)

    The distances are also available in ``sktime.dists_kernels.dtw``
    as pairwise transformers.

    Note that the more flexible options above may be less performant.

    DTW was originally proposed in [1]_, DTW computes the distance between two
    time series by considering their alignments during the calculation.
    This is done by measuring
    the pointwise distance (normally using Euclidean) between all elements of the two
    time series and then using dynamic programming to find the warping path
    that minimises the total pointwise distance between realigned series.

    DDTW is an adaptation of DTW originally proposed in [2]_. DDTW attempts to
    improve on dtw by better account for the 'shape' of the time series.
    This is done by considering y axis data points as higher level features of 'shape'.
    To do this the first derivative of the sequence is taken, and then using this
    derived sequence a dtw computation is done.

    WDTW was first proposed in [3]_, it adds a multiplicative weight penalty based on
    the warping distance. This means that time series with lower phase difference have
    a smaller weight imposed (i.e less penalty imposed) and time series with larger
    phase difference have a larger weight imposed (i.e. larger penalty imposed).

    WDDTW was first proposed in [3]_ as an extension of DDTW. By adding a weight
    to the derivative it means the alignment isn't only considering the shape of the
    time series, but also the phase.

    Parameters
    ----------
    weighted : bool, optional, default=False
        whether a weighted version of the distance is computed
        False = unmodified distance, i.e., dtw distance or derivative dtw distance
        True = weighted distance, i.e., weighted dtw or derivative weighted dtw
    derivative : bool, optional, default=False
        whether the distance or the derivative distance is computed
        False = unmodified distance, i.e., dtw distance or weighted dtw distance
        True = derivative distance, i.e., derivative dtw distance or derivative wdtw
    window: int, defaults = None
        Sakoe-Chiba window radius
        one of three mutually exclusive ways to specify bounding matrix
        if ``None``, does not use Sakoe-Chiba window
        if ``int``, uses Sakoe-Chiba lower bounding window with radius ``window``.
        If ``window`` is passed, ``itakura_max_slope`` will be ignored.
    itakura_max_slope: float, between 0. and 1., default = None
        Itakura parallelogram slope
        one of three mutually exclusive ways to specify bounding matrix
        if ``None``, does not use Itakura parallelogram lower bounding
        if ``float``, uses Itakura parallelogram lower bounding,
        with slope gradient ``itakura_max_slope``
    bounding_matrix: optional, 2D np.ndarray, default=None
        one of three mutually exclusive ways to specify bounding matrix
        must be of shape ``(len(X), len(X2))``, ``len`` meaning number time points,
        where ``X``, ``X2`` are the two time series passed in transform
        Custom bounding matrix to use.
        If provided, then ``window`` and ``itakura_max_slope`` are ignored.
        The matrix should be structured so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should
        be infinity.
    g: float, optional, default = 0. Used only if ``weighted=True``.
        Constant that controls the curvature (slope) of the function;
        that is, ``g`` controls the level of penalisation for the points
        with larger phase difference.

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    .. [2] Keogh, Eamonn & Pazzani, Michael. (2002). Derivative Dynamic Time Warping.
        First SIAM International Conference on Data Mining.
        1. 10.1137/1.9781611972719.1.
    .. [3] Young-Seon Jeong, Myong K. Jeong, Olufemi A. Omitaomu, Weighted dynamic time
    warping for time series classification, Pattern Recognition, Volume 44, Issue 9,
    2011, Pages 2231-2240, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2010.09.022.

    Examples
    --------
    >>> from sktime.utils._testing.series import _make_series
    >>> from sktime.alignment.dtw_numba import AlignerDtwNumba
    >>>
    >>> X0 = _make_series(return_mtype="pd.DataFrame")  # doctest: +SKIP
    >>> X1 = _make_series(return_mtype="pd.DataFrame")  # doctest: +SKIP
    >>> d = AlignerDtwNumba(weighted=True, derivative=True)  # doctest: +SKIP
    >>> align = d.fit([X0, X1]).get_alignment()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["chrisholder", "TonyBagnall", "fkiraly"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "capability:multiple-alignment": False,  # can align more than two sequences?
        "capability:distance": True,  # does compute/return overall distance?
        "capability:distance-matrix": True,  # does compute/return distance matrix?
        "capability:unequal_length": False,  # can align sequences of unequal length?
        "X_inner_mtype": "numpy3D",
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        weighted: bool = False,
        derivative: bool = False,
        window=None,
        itakura_max_slope=None,
        bounding_matrix: np.ndarray = None,
        g: float = 0.0,
    ):
        self.weighted = weighted
        self.derivative = derivative
        self.window = window
        self.itakura_max_slope = itakura_max_slope
        self.bounding_matrix = bounding_matrix
        self.g = g

        if not weighted and not derivative:
            metric_key = "dtw"
        elif not weighted and derivative:
            metric_key = "ddtw"
        elif weighted and not derivative:
            metric_key = "wdtw"
        elif weighted and derivative:
            metric_key = "wddtw"

        self.metric_key = metric_key

        kwargs = {
            "window": window,
            "itakura_max_slope": itakura_max_slope,
            "bounding_matrix": bounding_matrix,
        }

        # g is used only for weighted dtw
        if weighted:
            kwargs["g"] = g

        self.kwargs = kwargs

        super().__init__()

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
        from sktime.distances import distance_alignment_path

        X1 = X[0]
        X2 = X[1]

        metric_key = self.metric_key
        kwargs = self.kwargs

        path, dist = distance_alignment_path(X1, X2, metric=metric_key, **kwargs)
        self.path_ = path
        self.dist_ = dist

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
        path = self.path_
        ind0, ind1 = zip(*path)

        # convert to required data frame format and return
        aligndf = pd.DataFrame({"ind0": ind0, "ind1": ind1})

        return aligndf

    def _get_distance(self):
        """Return overall distance of alignment.

        Behaviour: returns overall distance corresponding to alignment
            not all aligners will return or implement this (optional)

        Returns
        -------
        distance: float - overall distance between all elements of X passed to fit
        """
        return self.dist_

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
        distmat[0, 1] = self.dist_
        distmat[1, 0] = self.dist_

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for aligners.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params0 = {}
        params1 = {"weighted": True}
        # derivative alignment paths do not seem to work - memouts. Bug?
        # params2 = {"derivative": True, "window": 0.2}
        # params3 = {"weighted": True, "derivative": True, "g": 0.05}

        return [params0, params1]
        # return [params0, params1, params2, params3]
