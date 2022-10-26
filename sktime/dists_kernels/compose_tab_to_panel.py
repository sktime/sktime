# -*- coding: utf-8 -*-
"""
Composers that create panel pairwise transformers from table pairwise transformers.

Currently implemented composers in this module:

    AggrDist - panel distance from aggregation of tabular distance matrix entries
    FlatDist - panel distance from applying tabular distance to flattened panel matrix
"""

__author__ = ["fkiraly"]

from deprecated.sphinx import deprecated

from sktime.dists_kernels.distances.compose_tab_to_panel import (
    AggrDist as new_aggr_class,
)
from sktime.dists_kernels.distances.compose_tab_to_panel import (
    FlatDist as new_flat_class,
)


# TODO: remove file in v0.15.0
@deprecated(
    version="0.13.4",
    reason="AggrDist has moved and this import will be removed in 0.15.0. Import from sktime.dists_kernels.distances",  # noqa: E501
    category=FutureWarning,
)
class AggrDist(new_aggr_class):
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
    """

    def __init__(
        self,
        transformer,
        aggfunc=None,
        aggfunc_is_symm=False,  # False for safety, but set True later if aggfunc=None
    ):
        super(AggrDist, self).__init__(
            transformer=transformer, aggfunc=aggfunc, aggfunc_is_symm=aggfunc_is_symm
        )


# TODO: remove file in v0.15.0
@deprecated(
    version="0.13.4",
    reason="FlatDist has moved and this import will be removed in 0.15.0. Import from sktime.dists_kernels.distances",  # noqa: E501
    category=FutureWarning,
)
class FlatDist(new_flat_class):
    r"""Panel distance from applying tabular distance to flattened time series.

    Applies the wrapped tabular distance to flattened series.
    Flattening is done to a 2D numpy array of shape (n_instances, (n_vars, n_timepts))

    Formal details (for real valued objects, mixed typed rows in analogy):
    Let :math:`d:\mathbb{R}^k \times \mathbb{R}^{k}\rightarrow \mathbb{R}`
    be the pairwise function in `transformer`, when applied to `k`-vectors.
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
    transformer: pairwise transformer of BasePairwiseTransformer scitype
    """

    def __init__(self, transformer):
        super(FlatDist, self).__init__(transformer=transformer)
