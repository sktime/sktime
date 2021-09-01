# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from scipy import sparse
from sklearn.pipeline import FeatureUnion as _FeatureUnion
from sklearn.pipeline import _fit_transform_one
from sklearn.pipeline import _transform_one

from sktime.transformations.base import _PanelToPanelTransformer

__all__ = ["FeatureUnion"]
__author__ = ["Markus Löning"]


class FeatureUnion(_FeatureUnion, _PanelToPanelTransformer):
    """Concatenates results of multiple transformer objects.
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.
    Parameters of the transformations may be set using its name and the
    parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer,
    or removed by setting to 'drop' or ``None``.
    Parameters
    ----------
    transformer_list : list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context.
        ``-1`` means using all processors.
    transformer_weights : dict, optional
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
    """

    _required_parameters = ["transformer_list"]

    def __init__(
        self,
        transformer_list,
        n_jobs=None,
        transformer_weights=None,
        preserve_dataframe=True,
    ):
        self.preserve_dataframe = preserve_dataframe
        super(FeatureUnion, self).__init__(
            transformer_list, n_jobs=n_jobs, transformer_weights=transformer_weights
        )

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformations, transform the data and concatenate results.
        Parameters
        ----------
        X : pandas DataFrame
            Input data to be transformed.
        y : pandas Series, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        Xt : pandas DataFrame
            hstack of results of transformations. sum_n_components is the
            sum of n_components (output dimension) over transformations.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter()
        )

        if not result:
            # All transformations are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        Xs = self._hstack(list(Xs))
        self._is_fitted = True
        return Xs

    def fit(self, X, y=None, **fit_params):
        super(FeatureUnion, self).fit(X, y, **fit_params)
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : pandas DataFrame
            Input data to be transformed.
        Returns
        -------
        Xt : pandas DataFrame
            hstack of results of transformations. sum_n_components is the
            sum of n_components (output dimension) over transformations.
        """
        self.check_is_fitted()
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter()
        )

        if not Xs:
            # All transformations are None
            return np.zeros((X.shape[0], 0))

        else:
            return self._hstack(list(Xs))

    def _hstack(self, Xs):
        """
        Stacks X horizontally.
        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames
        """

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()

        types = set(type(X) for X in Xs)
        if self.preserve_dataframe and (pd.Series in types or pd.DataFrame in types):
            return pd.concat(Xs, axis=1)

        else:
            return np.hstack(Xs)
