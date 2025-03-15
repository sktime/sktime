"""Wrappers to convert distance to kernel or kernel to distance."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels.base import BasePairwiseTransformerPanel

SUPPORTED_MTYPES = ["pd-multiindex", "nested_univ", "df-list", "numpy3D"]


def _trafo_diag(fun):
    """Obtain a function which returns the diagonal from one that returns a matrix."""

    def diag(X):
        mat = fun(X)
        return np.diag(mat)

    return diag


class KernelFromDist(BasePairwiseTransformerPanel):
    r"""Kernel function obtained from a distance function.

    Formal details (for real valued objects, mixed typed rows in analogy):
    Let :math:`d: \mathbb{R}^D \times \mathbb{R}^D\rightarrow \mathbb{R}`
    be the pairwise function in ``dist``, when applied to ``D``-vectors.
    If ``dist_diag=None``, then ``KernelFromDist(dist)`` corresponds to the kernel
    function
    :math:`k(x, y) := d(x, x)^2 + d(y, y)^2 - 0.5 \cdot d(x, y)^2`.
    If ``dist_diag`` is provided,
    and corresponds to a function :math:`f:\mathbb{R}^D \rightarrow \mathbb{R}`,
    then ``KernelFromDist(dist)`` corresponds to the kernel function
    :math:`k(x, y) := f(x, x)^2 + f(y, y)^2 - 0.5 \cdot d(x, y)^2`.

    It should be noted that :math:`k` is, in general, not positive semi-definite.

    Parameters
    ----------
    dist : pairwise transformer of BasePairwiseTransformer scitype, or
        callable np.ndarray (n_samples, nd) x (n_samples, nd) -> (n_samples x n_samples)
    dist_diag : pairwise transformer of BasePairwiseTransformer scitype, or
        series-to-panel transformer of Basetransformer scitype, or
        callable np.ndarray (n_samples, nd) -> (n_samples, )
    """

    _tags = {
        "authors": "fkiraly",
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
        "pwtrafo_type": "kernel",
    }

    def __init__(self, dist, dist_diag=None):
        self.dist = dist
        self.dist_diag = dist_diag

        super().__init__()

        # set property tags based on tags of components
        missing = True
        multi = True
        unequal = True
        if isinstance(dist, BasePairwiseTransformerPanel):
            missing = missing and dist.get_tag("capability:missing_values")
            multi = multi and dist.get_tag("capability:multivariate")
            unequal = unequal and dist.get_tag("capability:unequal_length")

        tag_dict = {
            "capability:missing_values": missing,
            "capability:multivariate": multi,
            "capability:unequal_length": unequal,
        }

        self.set_tags(**tag_dict)

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
        from sktime.transformations.base import BaseTransformer

        dist = self.dist
        dist_diag = self.dist_diag
        if dist_diag is None:
            dist_diag = dist

        if isinstance(dist_diag, BasePairwiseTransformerPanel):
            diagfun = dist_diag.transform_diag
        elif isinstance(dist_diag, BaseTransformer):
            diagfun = dist_diag.fit_transform
        else:
            diagfun = _trafo_diag(dist_diag)

        distmat = dist(X, X2) ** 2

        diag1 = diagfun(X)
        if X2 is None:
            diag2 = diag1
        else:
            diag2 = diagfun(X2)

        diag1 = np.array(diag1).flatten() ** 2
        diag2 = np.array(diag2).flatten() ** 2

        n, m = distmat.shape

        mat1 = np.tile(np.expand_dims(diag1, 1), m)
        mat2 = np.tile(np.expand_dims(diag2, 1), n)
        mat2 = mat2.transpose()

        kernmat = mat1 + mat2 - 0.5 * distmat

        return kernmat

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
        from sktime.transformations.series.adapt import PandasTransformAdaptor
        from sktime.transformations.series.summarize import SummaryTransformer

        params1 = {"dist": DtwDist()}
        t = SummaryTransformer("mean", None)
        # we need this since multivariate summary produces two columns
        # if one column, has no effect; if multiple, takes means by row
        t = PandasTransformAdaptor("mean", {"axis": 1}) * t
        params2 = {"dist": DtwDist(), "dist_diag": t}

        return [params1, params2]


class DistFromKernel(BasePairwiseTransformerPanel):
    r"""Distance function obtained from a kernel function.

    Formal details (for real valued objects, mixed typed rows in analogy):
    Let :math:`k: \mathbb{R}^D \times \mathbb{R}^D\rightarrow \mathbb{R}`
    be the pairwise function in ``kernel``, when applied to ``D``-vectors.
    ``DistFromKernel(dist)`` corresponds to the distance function
    :math:`d(x, y) := \sqrt{k(x, x) + k(y, y) - 2 \cdot k(x, y)}`.

    It should be noted that if :math:`k` is positive semi-definite,
    then :math:`d` will be a metric and satisfy the triangle inequality.

    Parameters
    ----------
    kernel : pairwise transformer of BasePairwiseTransformer scitype, or
        callable np.ndarray (n_samples, nd) x (n_samples, nd) -> (n_samples x n_samples)
    """

    _tags = {
        "authors": "fkiraly",
        "X_inner_mtype": SUPPORTED_MTYPES,
        "capability:missing_values": True,  # can estimator handle missing data?
        "capability:multivariate": True,  # can estimator handle multivariate data?
        "capability:unequal_length": True,  # can dist handle unequal length panels?
        "pwtrafo_type": "distance",
    }

    def __init__(self, kernel):
        self.kernel = kernel

        super().__init__()

        # set property tags based on tags of components
        tags_to_clone = [
            "capability:missing_values",
            "capability:multivariate",
            "capability:unequal_length",
        ]

        if isinstance(kernel, BasePairwiseTransformerPanel):
            self.clone_tags(kernel, tags_to_clone)

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
        kernel = self.kernel

        kernelmat = kernel(X, X2)

        if isinstance(kernel, BasePairwiseTransformerPanel):
            diagfun = kernel.transform_diag
        else:
            diagfun = _trafo_diag(kernel)

        diag1 = diagfun(X)
        if X2 is None:
            diag2 = diag1
        else:
            diag2 = diagfun(X2)

        diag1 = np.array(diag1).flatten()
        diag2 = np.array(diag2).flatten()

        n, m = kernelmat.shape

        mat1 = np.tile(np.expand_dims(diag1, 1), m)
        mat2 = np.tile(np.expand_dims(diag2, 1), n)
        mat2 = mat2.transpose()

        distmat = mat1 + mat2 - 2 * kernelmat
        distmat = np.sqrt(np.abs(distmat))

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
        from sktime.dists_kernels import DtwDist, EditDist

        params1 = {"kernel": DtwDist()}
        params2 = {"kernel": EditDist()}

        return [params1, params2]
