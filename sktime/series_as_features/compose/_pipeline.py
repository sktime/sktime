# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import clone
from sklearn.pipeline import FeatureUnion as _FeatureUnion

from sktime.transformations.base import _PanelToPanelTransformer

__all__ = ["FeatureUnion"]
__author__ = ["Markus Löning"]


class FeatureUnion(_FeatureUnion, _PanelToPanelTransformer):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.
    Parameters of the transformations may be set using its name and the
    parameter name separated by a '__'. A transformer may be replaced entirely by
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
    preserve_dataframe : bool
        Save constructed dataframe.
    """

    _required_parameters = ["transformers"]

    def __init__(
        self,
        transformers,
        n_jobs=None,
        transformer_weights=None,
        preserve_dataframe=True,
    ):

        self.transformers = transformers
        self.preserve_dataframe = preserve_dataframe

        transformer_list = []
        for x in transformers:
            transformer_list += [(x[0], clone(x[1]))]
        super(FeatureUnion, self).__init__(
            transformer_list, n_jobs=n_jobs, transformer_weights=transformer_weights
        )

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit(self, X, y=None, **fit_params):
        """Fit parameters."""
        super().fit(X, y, **fit_params)
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results."""
        self.check_is_fitted()
        return super().transform(X)

    def fit_transform(self, X, y, **fit_params):
        """Transform X separately by each transformer, concatenate results."""
        return self.fit(X, y, **fit_params).transform(X)

    def _hstack(self, Xs):
        """
        Stacks X horizontally.

        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames.
        """
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()

        types = {type(X) for X in Xs}
        if self.preserve_dataframe and (pd.Series in types or pd.DataFrame in types):
            return pd.concat(Xs, axis=1)

        else:
            return np.hstack(Xs)

    @classmethod
    def get_test_params(cls):
        """Test parameters for FeatureUnion."""
        from sklearn.preprocessing import StandardScaler

        SERIES_TO_SERIES_TRANSFORMER = StandardScaler()

        from sktime.transformations.panel.compose import SeriesToSeriesRowTransformer

        TRANSFORMERS = [
            (
                "transformer1",
                SeriesToSeriesRowTransformer(
                    SERIES_TO_SERIES_TRANSFORMER, check_transformer=False
                ),
            ),
            (
                "transformer2",
                SeriesToSeriesRowTransformer(
                    SERIES_TO_SERIES_TRANSFORMER, check_transformer=False
                ),
            ),
        ]

        return {"transformers": TRANSFORMERS}
