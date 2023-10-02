# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Lucky dynamic time warping distance."""

from sktime.dists_kernels.base._delegate import _DelegatedPairwiseTransformerPanel


class LuckyDtwDist(_DelegatedPairwiseTransformerPanel):
    """Lucky dynamic time warping distance.

    Implements lucky dynamic time warping distance [1]_.
    Uses Euclidean distance for multivariate data.

    Based on code by Krisztian A Buza's research group.

    Parameters
    ----------
    window: int, optional (default=None)
        Maximum distance between indices of aligned series, aka warping window.
        If None, defaults to max(len(ts1), len(ts2)), i.e., no warping window.

    References
    ----------
    ..[1] Stephan Spiegel, Brijnesh-Johannes Jain, and Sahin Albayrak.
        Fast time series classification under lucky time warping distance.
        Proceedings of the 29th Annual ACM Symposium on Applied Computing. 2014.
    """

    _tags = {
        "symmetric": True,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
        "pwtrafo_type": "distance",  # type of pw. transformer, "kernel" or "distance"
    }

    def __init__(self, window=None):
        super().__init__()

        from sktime.alignment.lucky import AlignerLuckyDtw
        from sktime.dists_kernels.compose_from_align import DistFromAligner

        self.estimator_ = DistFromAligner(AlignerLuckyDtw(window=window))
