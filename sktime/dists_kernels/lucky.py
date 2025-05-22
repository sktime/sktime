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
        # packaging info
        # --------------
        "authors": ["fkiraly", "Kristian A Buza"],
        # estimator type
        # --------------
        "symmetric": True,  # is the transformer symmetric, i.e., t(x,y)=t(y,x) always?
        "capability:missing_values": False,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
        "pwtrafo_type": "distance",  # type of pw. transformer, "kernel" or "distance"
    }

    def __init__(self, window=None):
        self.window = window

        super().__init__()

        from sktime.alignment.lucky import AlignerLuckyDtw
        from sktime.dists_kernels.compose_from_align import DistFromAligner

        self.estimator_ = DistFromAligner(AlignerLuckyDtw(window=window))

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
        params1 = {"window": 4}

        return [params0, params1]
