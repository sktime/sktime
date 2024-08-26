"""BaseEstimator interface to sktime dtw distances in distances module."""

__author__ = ["fkiraly"]

from typing import Union

import numpy as np

from sktime.distances import pairwise_distance
from sktime.dists_kernels.base import BasePairwiseTransformerPanel


class DtwDist(BasePairwiseTransformerPanel):
    r"""Interface to sktime native dtw distances, with derivative or weighting.

    Interface to simple dynamic time warping (DTW) distance,
    and the following weighted/derivative versions:

    * WDTW - weighted dynamic tyme warping - ``weighted=True, derivative=False`
    * DDTW - derivative dynamic time warping - ``weighted=False, derivative=True``
    * WDDTW - weighted derivative dynamic time
      warping - ``weighted=True, derivative=True``

    ``sktime`` interface to the efficient ``numba`` implementations
    provided by ``pairwise_distance`` in ``sktime.distances``.

    This estimator provides performant implementation of time warping distances for:
    * time series of equal length
    * the Euclidean pairwise distance

    For unequal length time series, use ``sktime.dists_kernels.DistFromAligner``
    with a time warping aligner such as ``sktime.aligners.AlignerDTW``.
    To use arbitrary pairwise distances, use ``sktime.aligners.AlignerDTWfromDist``.
    (for derivative DTW, pipeline an alignment distance with ``Differencer``)

    Note that the more flexible options above may be less performant.

    The algorithms are also available as alignment estimators
    ``sktime.alignment.dtw_numba``, producing alignments aka alignment paths.

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
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.dists_kernels.dtw import DtwDist
    >>>
    >>> X, _ = load_unit_test(return_type="pd-multiindex")  # doctest: +SKIP
    >>> d = DtwDist(weighted=True, derivative=True)  # doctest: +SKIP
    >>> distmat = d.transform(X)  # doctest: +SKIP

    distances are also callable, this does the same:

    >>> distmat = d(X)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["chrisholder", "TonyBagnall", "fkiraly"],
        "python_dependencies": "numba",
        # estimator type
        # --------------
        "symmetric": True,  # all the distances are symmetric
        "X_inner_mtype": "numpy3D",
        "capability:unequal_length": False,  # can dist handle unequal length panels?
    }

    def __init__(
        self,
        weighted: bool = False,
        derivative: bool = False,
        window: Union[int, None] = None,
        itakura_max_slope: Union[float, None] = None,
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

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

            Core logic

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: 3D np.array of shape [num_instances, num_vars, num_time_points]
        X2: 3D np.array of shape [num_instances, num_vars, num_time_points], optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]
        """
        metric_key = self.metric_key
        kwargs = self.kwargs

        distmat = pairwise_distance(X, X2, metric=metric_key, **kwargs)

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for distance/kernel transformers.

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
        params2 = {"derivative": True, "window": 0.2}
        params3 = {"weighted": True, "derivative": True, "g": 0.05}

        return [params0, params1, params2, params3]
