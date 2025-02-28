"""Feature union."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly", "mloning"]
__all__ = ["FeatureUnion"]

import pandas as pd

from sktime.base._meta import _HeterogenousMetaEstimator
from sktime.transformations.base import BaseTransformer
from sktime.utils.multiindex import flatten_multiindex


class FeatureUnion(_HeterogenousMetaEstimator, BaseTransformer):
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
    flatten_transform_index : bool, optional (default=True)
        if True, columns of return DataFrame are flat, by "transformer__variablename"
        if False, columns are MultiIndex (transformer, variablename)
        has no effect if return mtype is one without column names
    """

    _tags = {
        "authors": ["fkiraly", "mloning"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": False,  # depends on components
        "univariate-only": False,  # depends on components
        "handles-missing-data": False,  # depends on components
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": "None",
        "X-y-must-have-same-index": False,
        "enforce_index_type": None,
        "fit_is_empty": False,
        "transform-returns-same-time-index": False,
        "skip-inverse-transform": False,
        "capability:inverse_transform": False,
        "visual_block_kind": "parallel",
        # unclear what inverse transform should be, since multiple inverse_transform
        #   would have to inverse transform to one
    }

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator) pairs for the default
    _steps_attr = "_transformer_list"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    _steps_fitted_attr = "transformer_list_"

    def __init__(
        self,
        transformer_list,
        n_jobs=None,
        transformer_weights=None,
        flatten_transform_index=True,
    ):
        self.transformer_list = transformer_list
        transformer_list_ = self._check_estimators(
            transformer_list, cls_type=BaseTransformer
        )
        self.transformer_list_ = transformer_list_

        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.flatten_transform_index = flatten_transform_index

        super().__init__()

        t_outs = [t.get_tag("scitype:transform-output") for _, t in transformer_list_]
        t_ins = [t.get_tag("scitype:transform-input") for _, t in transformer_list_]
        # todo: error or special case handling if these are not all the same
        self.set_tags(**{"scitype:transform-output": t_outs[0]})
        self.set_tags(**{"scitype:transform-input": t_ins[0]})

        # abbreviate for readability
        ests = self.transformer_list_

        # set property tags based on tags of components
        self._anytag_notnone_set("y_inner_mtype", ests)
        self._anytag_notnone_set("scitype:transform-labels", ests)

        self._anytagis_then_set("scitype:instancewise", False, True, ests)
        self._anytagis_then_set("X-y-must-have-same-index", True, False, ests)
        self._anytagis_then_set("fit_is_empty", False, True, ests)
        self._anytagis_then_set("transform-returns-same-time-index", False, True, ests)
        self._anytagis_then_set("skip-inverse-transform", True, False, ests)
        # self._anytagis_then_set("capability:inverse_transform", False, True, ests)
        self._anytagis_then_set("handles-missing-data", False, True, ests)
        self._anytagis_then_set("univariate-only", True, False, ests)

        # if any of the components require_X or require_y, set it for the composite
        self._anytagis_then_set("requires_X", True, False, ests)
        self._anytagis_then_set("requires_y", True, False, ests)

    @property
    def _transformer_list(self):
        return self._get_estimator_tuples(self.transformer_list, clone_ests=False)

    @_transformer_list.setter
    def _transformer_list(self, value):
        self.transformer_list = value
        self.transformer_list_ = self._check_estimators(value, cls_type=BaseTransformer)

    def __add__(self, other):
        """Magic + method, return (right) concatenated FeatureUnion.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of ``self`` (first) with ``other``
        (last).
            not nested, contains only non-FeatureUnion ``sktime`` transformers
        """
        from sktime.registry import coerce_scitype

        other = coerce_scitype(other, "transformer")
        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=FeatureUnion,
            attr_name="transformer_list",
            concat_order="left",
        )

    def __radd__(self, other):
        """Magic + method, return (left) concatenated FeatureUnion.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of ``self`` (last) with ``other``
        (first).
            not nested, contains only non-FeatureUnion ``sktime`` transformers
        """
        from sktime.registry import coerce_scitype

        other = coerce_scitype(other, "transformer")
        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=FeatureUnion,
            attr_name="transformer_list",
            concat_order="right",
        )

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame, Series, Panel, or Hierarchical mtype format
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        self.transformer_list_ = self._check_estimators(
            self.transformer_list, cls_type=BaseTransformer
        )

        for _, transformer in self.transformer_list_:
            transformer.fit(X=X, y=y)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame, Series, Panel, or Hierarchical mtype format
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        # retrieve fitted transformers, apply to the new data individually
        transformers = self._get_estimator_list(self.transformer_list_)
        if not self.get_tag("fit_is_empty", False):
            Xt_list = [trafo.transform(X, y) for trafo in transformers]
        else:
            Xt_list = [trafo.fit_transform(X, y) for trafo in transformers]

        transformer_names = self._get_estimator_names(self.transformer_list_)

        Xt = pd.concat(
            Xt_list, axis=1, keys=transformer_names, names=["transformer", "variable"]
        )

        if self.flatten_transform_index:
            Xt.columns = flatten_multiindex(Xt.columns)

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for FeatureUnion."""
        from sktime.transformations.series.boxcox import BoxCoxTransformer
        from sktime.transformations.series.exponent import ExponentTransformer

        # with name and estimator tuple, all transformers don't have fit
        TRANSFORMERS = [
            ("transformer1", ExponentTransformer(power=4)),
            ("transformer2", ExponentTransformer(power=0.25)),
        ]
        params1 = {"transformer_list": TRANSFORMERS}

        # only with estimators, some transformers have fit, some not
        params2 = {
            "transformer_list": [
                ExponentTransformer(power=4),
                ExponentTransformer(power=0.25),
                BoxCoxTransformer(),
            ]
        }

        return [params1, params2]
