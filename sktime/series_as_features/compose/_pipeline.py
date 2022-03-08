# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations.base import BaseTransformer

__all__ = ["FeatureUnion"]
__author__ = ["mloning, fkiraly"]


class FeatureUnion(BaseTransformer, _HeterogenousMetaEstimator):
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
    preserve_dataframe : bool - deprecated
    flatten_transform_index : bool, optional (default=True)
        if True, columns of return DataFrame are flat, by "transformer__variablename"
        if False, columns are MultiIndex (transformer, variablename)
        has no effect if return mtypes is one without column names
    """

    _required_parameters = ["transformer_list"]

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": False,  # depends on components
        "univariate-only": False,  # depends on components
        "handles-missing-data": False,  # depends on components
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex"],
        "y_inner_mtype": "None",
        "X-y-must-have-same-index": False,
        "enforce_index_type": None,
        "fit-in-transform": False,
        "transform-returns-same-time-index": False,
        "skip-inverse-transform": False,
    }

    def __init__(
        self,
        transformer_list,
        n_jobs=None,
        transformer_weights=None,
        preserve_dataframe=True,
        flatten_transform_index=True,
    ):
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.preserve_dataframe = preserve_dataframe
        self.transformer_list = transformer_list
        self.flatten_transform_index = flatten_transform_index

        super(FeatureUnion, self).__init__()

    def _fit(self, X, y=None):
        """Fit parameters."""
        transformer_list = self.transformer_list

        # clone and fit transformers in transformer_list, store fitted copies
        transformer_list_ = [clone(trafo[1]).fit(X, y) for trafo in transformer_list]
        self.transformer_list_ = transformer_list_

        return self

    def _transform(self, X, y=None):
        """Transform X separately by each transformer, concatenate results."""
        # retrieve fitted transformers, apply to the new data individually
        transformer_list_ = self.transformer_list_
        Xt_list = [trafo.transform(X, y) for trafo in transformer_list_]

        transformer_names = [x[0] for x in self.transformer_list]

        Xt = pd.concat(
            Xt_list, axis=1, keys=transformer_names, names=["transformer", "variable"]
        )

        if self.flatten_transform_index:
            flat_index = pd.Index("__".join(x) for x in Xt.columns)
            Xt.columns = flat_index

        return Xt

    def get_params(self, deep=True):
        """Get parameters of estimator in `_forecasters`.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("transformer_list", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `_forecasters`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params("transformer_list", **kwargs)
        return self

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

        return {"transformer_list": TRANSFORMERS}
