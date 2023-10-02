"""Interface module to scipy.

Interface module to scipy.spatial's pairwise distance function cdist     exposes
parameters as scikit-learn hyper-parameters
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sktime.dists_kernels.base import BasePairwiseTransformer


class ScipyDist(BasePairwiseTransformer):
    """Interface to scipy distances.

    computes pairwise distances using scipy.spatial.distance.cdist
        includes Euclidean distance and p-norm (Minkowski) distance
            note: weighted distances are not supported

    Parameters
    ----------
    metric : string or function, as in cdist; default = 'euclidean'
        if string, one of: 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
            'jensenshannon',
            'kulsinski' (< scipy 1.11) or 'kulczynski1' (from scipy 1.11),
            'mahalanobis', 'matching', 'minkowski',
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

    _tags = {
        "symmetric": True,  # all the distances are symmetric
    }

    def __init__(
        self,
        metric="euclidean",
        p=2,
        colalign="intersect",
        var_weights=None,
        metric_kwargs=None,
    ):
        self.metric = metric
        self.p = p
        self.colalign = colalign
        self.var_weights = var_weights
        self.metric_kwargs = metric_kwargs

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
        X: pd.DataFrame of length n, or 2D np.array with n rows
        X2: pd.DataFrame of length m, or 2D np.array with m rows, optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X.iloc[i] and X2.iloc[j]
        """
        p = self.p
        metric = self.metric
        var_weights = self.var_weights
        metric_kwargs = self.metric_kwargs
        if metric_kwargs is None:
            metric_kwargs = {}

        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes("number").to_numpy(dtype="float")

        if isinstance(X2, pd.DataFrame):
            X2 = X2.select_dtypes("number").to_numpy(dtype="float")

        if np.ndim(var_weights) == 1:
            if len(var_weights) == len(X.columns) == len(X2.columns):
                X = var_weights * X
                X2 = var_weights * X2
            else:
                raise ValueError(
                    "weights vector length must be equal to X and X2 number of columns"
                )

        if metric == "minkowski" and "p" not in metric_kwargs.keys():
            distmat = cdist(XA=X, XB=X2, metric=metric, p=p, **metric_kwargs)
        else:
            distmat = cdist(XA=X, XB=X2, metric=metric, **metric_kwargs)

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for distance/kernel transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # default settings
        params1 = {}

        # using kwargs
        params2 = {"metric": "minkowski", "p": 3}

        return [params1, params2]
