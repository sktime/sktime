# -*- coding: utf-8 -*-
"""Interface module to scipy.

Interface module to scipy.spatial's pairwise distance function cdist
    exposes parameters as scikit-learn hyper-parameters
"""

__author__ = ["fkiraly"]


from deprecated.sphinx import deprecated

from sktime.dists_kernels.distances.scipy_dist import ScipyDist as new_class


# TODO: remove file in v0.15.0
@deprecated(
    version="0.13.4",
    reason="ScipyDist has moved and this import will be removed in 0.15.0. Import from sktime.dists_kernels.distances",  # noqa: E501
    category=FutureWarning,
)
class ScipyDist(new_class):
    """Interface to scipy distances.

    computes pairwise distances using scipy.spatial.distance.cdist
        includes Euclidean distance and p-norm (Minkowski) distance
            note: weighted distances are not supported

    Parameters
    ----------
    metric : string or function, as in cdist; default = 'euclidean'
        if string, one of: 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
            'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
            'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
            'sokalsneath', 'sqeuclidean', 'yule'
        if function, should have signature 1D-np.array x 1D-np.array -> float
    p:  if metric='minkowski', the "p" in "p-norm", otherwise irrelevant
    colalign : string, one of 'intersect' (default), 'force-align', 'none'
        controls column alignment if X, X2 passed in fit are pd.DataFrame
        columns between X and X2 are aligned via column names
        if 'intersect', distance is computed on columns occurring both in X and X2,
            other columns are discarded; column ordering in X2 is copied from X
        if 'force-align', raises an error if the set of columns in X, X2 differs;
            column ordering in X2 is copied from X
        if 'none', X and X2 are passed through unmodified (no columns are aligned)
            note: this will potentially align "non-matching" columns
    var_weights : 1D np.array of float or None, default=None
        weight/scaling vector applied to variables in X/X2
        before being passed to cdist, i-th col of X/X2 is multiplied by var_weights[i]
        if None, equivalent to all-ones vector
    metric_kwargs : dict, optional, default=None
        any kwargs passed to the metric in addition, i.e., to the function cdist
        common kwargs: "w" : array-like, same length as X.columns, weights for metric
        refer to scipy.spatial.distance.dist for a documentation of other extra kwargs
    """

    def __init__(
        self,
        metric="euclidean",
        p=2,
        colalign="intersect",
        var_weights=None,
        metric_kwargs=None,
    ):
        super(ScipyDist, self).__init__(
            metric=metric,
            p=p,
            colalign=colalign,
            var_weights=var_weights,
            metric_kwargs=metric_kwargs,
        )
