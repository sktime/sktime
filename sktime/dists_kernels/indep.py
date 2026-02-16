"""Wrappers to convert distance to kernel or kernel to distance."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels.base import BasePairwiseTransformerPanel

SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ"]


class IndepDist(BasePairwiseTransformerPanel):
    r"""Variable-wise aggregate of multivariate kernel or distance function.

    A common baseline method to turn a univariate time series distance or kernel
    into a multivariate time series distance or kernel.

    Also sometimes known as "independent distance" in the special case where ``aggfun``
    is the sum or mean and the pairwise transformer is a time series distance.

    Formal details (for real valued objects, mixed typed rows in analogy):
    Let :math:`d: \mathbb{R}^n \times \mathbb{R}^n\rightarrow \mathbb{R}`
    be the pairwise function in ``dist``, when applied to univariate series of length
    :math:`n`.
    This class represents the pairwise function
    :math:`d_g: \mathbb{R}^{n\times D} \times \mathbb{R}^{n\times D}\rightarrow
    \mathbb{R}`
    defined as :math:`d_g(x, y) := g(d(x_1, y_1), \dots, d(x_D, y_D))`,
    where :math:`x_i`, :math:`y_i` denote the :math:`i`-th column,
    and :math:`x`, ``:math:``y` are interpreted as multivariate time series with
    :math:`D` variables, and where :math:`g` is a function
    :math:`g: \mathbb{R}^D\times \mathbb{R}^D \rightarrow \mathbb{R}`,
    representing the input ``aggfun``.

    In particular, if ``aggfun="sum"`` (or default), then
    :math:`g(x) = \sum_{i=1}^D x_i`, and
    :math:`d_g(x, y) := \sum_{i=1}^D d(x_i, y_i)`,
    which corresponds to the usual terminology "independent distance".

    Parameters
    ----------
    dist : pairwise transformer of BasePairwiseTransformer scitype, or
        callable np.ndarray (n_samples, nd) x (n_samples, nd) -> (n_samples x n_samples)
    aggfun : optional, str or callable np.ndarray (m, nd, nd) -> (nd, nd)
        aggregation function over the variables, :math:`g` above
        "sum" = np.sum = default
        "mean" = np.mean
        "median" = np.median
        "max" = np.max
        "min" = np.min
        when starting with a function (m) -> scalar, use np.apply_along_axis
        to create a function (m, nd, nd) -> (nd, nd) and pass that as ``aggfun``

    Examples
    --------
    >>> from sktime.dists_kernels.indep import IndepDist
    >>> from sktime.dists_kernels.dtw import DtwDist
    >>>
    >>> dist = IndepDist(DtwDist())
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": "fkiraly",
        # estimator type
        # --------------
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, dist, aggfun=None):
        self.dist = dist
        self.aggfun = aggfun

        super().__init__()

        # set property tags based on tags of components
        missing = True
        unequal = True
        if isinstance(dist, BasePairwiseTransformerPanel):
            missing = missing and dist.get_tag("capability:missing_values")
            unequal = unequal and dist.get_tag("capability:unequal_length")
            pw_type = unequal = unequal and dist.get_tag("pwtrafo_type")

        tag_dict = {
            "capability:missing_values": missing,
            "capability:unequal_length": unequal,
            "pwtrafo_type": pw_type,
        }

        self.set_tags(**tag_dict)

        aggfun_dict = {
            "mean": np.mean,
            "sum": np.sum,
            "max": np.max,
            "min": np.min,
            "median": np.median,
        }
        if aggfun is None:
            self._aggfun = np.mean
        elif isinstance(aggfun, str):
            if aggfun not in aggfun_dict.keys():
                msg = (
                    f"error in IndepDist, aggfun must be callable or one of the "
                    f"strings {aggfun_dict.keys()}, but found {aggfun}"
                )
                raise ValueError(msg)
            self._aggfun = aggfun_dict[aggfun]
        else:
            self._aggfun = aggfun

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        private _transform containing core logic, called from public transform

        Parameters
        ----------
        X: sktime Panel data container
        X2: sktime Panel data container

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        dist = self.dist
        aggfun = self._aggfun

        mats = []
        for col in X.columns:
            X_sub = X.loc[:, [col]]
            if X2 is None:
                X2_sub = None
            else:
                X2_sub = X2.loc[:, [col]]
            mats += [dist.transform(X_sub, X2_sub)]

        if isinstance(self.aggfun, str) or self.aggfun is None:
            distmat = aggfun(mats, axis=0)
        else:
            distmat = aggfun(mats)
        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sktime.dists_kernels.dtw import DtwDist

        params1 = {"dist": DtwDist()}
        params2 = {"dist": DtwDist(), "aggfun": "median"}
        params3 = {"dist": DtwDist(), "aggfun": _testfun}

        return [params1, params2, params3]


def _testfun(x):
    return np.mean(x, axis=0)
