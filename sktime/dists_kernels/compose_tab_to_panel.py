# -*- coding: utf-8 -*-
"""
Composers that create panel pairwise transformers from table pairwise transformers.

Currently implemented composers in this module:

    AggrDist - panel distance from aggregation of tabular distance matrix entries
    FlatDist - panel distance from applying tabular distance to flattened panel matrix
"""

__author__ = ["fkiraly"]

import numpy as np

from sktime.dists_kernels._base import (
    BasePairwiseTransformer,
    BasePairwiseTransformerPanel,
)
from sktime.utils._testing.deep_equals import deep_equals


class AggrDist(BasePairwiseTransformerPanel):
    r"""Panel distance from tabular distance aggregation.

    panel distance obtained by applying aggregation function to tabular distance matrix
        example: AggrDist(ScipyDist()) is mean Euclidean distance between series

    Formal details (for real valued objects, mixed typed rows in analogy):
    Let :math:`d: \mathbb{R}^k \times \mathbb{R}^{k}\rightarrow \mathbb{R}`
    be the pairwise function in `transformer`, when applied to `k`-vectors.
    Let :math:`f:\mathbb{R}^{n \ times m}` be the function `aggfunc` when applied to
    an :math:`(n \times m)` matrix.
    Let :math:`x_1, \dots, x_N\in \mathbb{R}^{n \times k}`,
    :math:`y_1, \dots y_M \in \mathbb{R}^{m \times k}` be collections of matrices,
    representing time series panel valued inputs `X` and `X2`, as follows:
    :math:`x_i` is the `i`-th instance in `X`, and :math:`x_{i, j\ell}` is the
    `j`-th time point, `\ell`-th variable of `X`. Analogous for :math:`y` and `X2`.

    Then, `transform(X, X2)` returns the :math:`(N \times M)` matrix
    with :math:`(i, j)`-th entry :math:`f \left((d(x_{i, a}, y_{j, b}))_{a, b}\right)`,
    where :math:`x_{i, a}` denotes the :math:`a`-th row of :math:`x_i`, and
    :math:`y_{j, b}` denotes the :math:`b`-th row of :math:`x_j`.

    Parameters
    ----------
    transformer: pairwise transformer of BasePairwiseTransformer scitype
    aggfunc: aggregation function (2D np.array) -> float or None, optional
        default = None = np.mean
    aggfunc_is_symm: bool, optional, default=False
        whether aggregation function is symmetric (should be set according to aggfunc)
            i.e., invariant under transposing argument, it always holds that
                aggfunc(matrix) = aggfunc(np.transpose(matrix))
            used for fast computation of the resultant matrix (if symmetric)
            if unknown, False is the "safe" option that ensures correctness

    Examples
    --------
    Mean pairwise euclidean distance between between time series
    >>> from sktime.dists_kernels import AggrDist, ScipyDist
    >>> mean_euc_tsdist = AggrDist(ScipyDist())

    Mean pairwise Gaussian kernel between time series
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> mean_gaussian_tskernel = AggrDist(RBF())
    """

    def __init__(
        self,
        transformer,
        aggfunc=None,
        aggfunc_is_symm=False,  # False for safety, but set True later if aggfunc=None
    ):

        self.aggfunc = aggfunc
        self.aggfunc_is_symm = aggfunc_is_symm
        self.transformer = transformer

        super(AggrDist, self).__init__()

        if self.aggfunc_is_symm:
            self.set_tags(**{"symmetric": True})

        if isinstance(transformer, BasePairwiseTransformer):
            tags_to_clone = ["capability:missing_values"]
            self.clone_tags(transformer, tags_to_clone)

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

            Core logic.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        n = len(X)
        m = len(X2)

        X_equals_X2 = deep_equals(X, X2)

        aggfunc = self.aggfunc
        aggfunc_is_symm = self.aggfunc_is_symm
        transformer = self.transformer
        if aggfunc is None:
            aggfunc = np.mean
            aggfunc_is_symm = True

        if isinstance(transformer, BasePairwiseTransformer):
            transformer_symm = transformer.get_tag("symmetric", False)
        else:
            transformer_symm = False

        # whether we know that resulting matrix must be symmetric
        # a sufficient condition for this:
        # transformer is symmetric; X equals X2; and aggfunc is symmetric
        all_symm = aggfunc_is_symm and transformer_symm and X_equals_X2

        distmat = np.zeros((n, m), dtype="float")

        for i in range(n):
            for j in range(m):

                if all_symm and j < i:
                    distmat[i, j] = distmat[j, i]
                else:
                    distmat[i, j] = aggfunc(self.transformer(X[i], X2[j]))

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for AggrDist."""
        from sklearn.gaussian_process.kernels import RBF

        from sktime.dists_kernels import ScipyDist

        params1 = {"transformer": ScipyDist(), "aggfunc_is_symm": True}
        params2 = {"transformer": ScipyDist(), "aggfunc_is_symm": False}

        # using callable from sklearn
        params3 = {"transformer": RBF()}

        return [params1, params2, params3]


class FlatDist(BasePairwiseTransformerPanel):
    r"""Panel distance or kernel from applying tabular trafo to flattened time series.

    Applies the wrapped tabular distance or kernel to flattened series.
    Flattening is done to a 2D numpy array of shape (n_instances, (n_vars, n_timepts))

    Formal details (for real valued objects, mixed typed rows in analogy):
    Let :math:`d:\mathbb{R}^k \times \mathbb{R}^{k}\rightarrow \mathbb{R}`
    be the pairwise function in `transformer`, when applied to `k`-vectors
    (here, :math:`d` could be a distance function or a kernel function).
    Let :math:`x_1, \dots, x_N\in \mathbb{R}^{n \times \ell}`,
    :math:`y_1, \dots y_M \in \mathbb{R}^{n \times \ell}` be collections of matrices,
    representing time series panel valued inputs `X` and `X2`, as follows:
    :math:`x_i` is the `i`-th instance in `X`, and :math:`x_{i, j\ell}` is the
    `j`-th time point, `\ell`-th variable of `X`. Analogous for :math:`y` and `X2`.
    Let :math:`f:\mathbb{R}^{n \times \ell} \rightarrow \mathbb{R}^{n \cdot \ell}`
    be the mapping that flattens matrices by column-first lexicographical ordering,
    and assume :math:`k = n \cdot \ell`.

    Then, `transform(X, X2)` returns the :math:`(N \times M)` matrix
    with :math:`(i, j)`-th entry :math:`d\left(f(x_i), f(y_j)\right)`.

    Parameters
    ----------
    transformer: pairwise transformer of BasePairwiseTransformer scitype, or
        callable np.ndarray (n_samples, d) x (n_samples, d) -> (n_samples x n_samples)

    Examples
    --------
    Euclidean distance between time series of equal length, considered as vectors
    >>> from sktime.dists_kernels import FlatDist, ScipyDist
    >>> euc_tsdist = FlatDist(ScipyDist())

    Gaussian kernel between time series of equal length, considered as vectors
    >>> from sklearn.gaussian_process.kernels import RBF
    >>> flat_gaussian_tskernel = FlatDist(RBF())
    """

    _tags = {
        "X_inner_mtype": "numpy3D",  # which mtype is used internally in _transform?
        "capability:unequal_length": False,
    }

    def __init__(self, transformer):

        self.transformer = transformer

        super(FlatDist, self).__init__()

        if isinstance(transformer, BasePairwiseTransformer):
            tags_to_clone = ["capability:missing_values"]
            self.clone_tags(transformer, tags_to_clone)

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

            Core logic.

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        n_inst, n_vars, n_tts = X.shape
        X = X.reshape(n_inst, n_vars * n_tts)

        n_inst2, n_vars2, n_tts2 = X2.shape
        X2 = X2.reshape(n_inst2, n_vars2 * n_tts2)

        if deep_equals(X, X2):
            return self.transformer(X)
        else:
            return self.transformer(X, X2)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for FlatDist."""
        from sklearn.gaussian_process.kernels import RBF

        from sktime.dists_kernels import ScipyDist

        # using sktime pairwise transformer
        params1 = {"transformer": ScipyDist()}

        # using callable from sklearn
        params2 = {"transformer": RBF()}

        return [params1, params2]
