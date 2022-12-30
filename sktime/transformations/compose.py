# -*- coding: utf-8 -*-
"""Meta-transformers for building composite transformers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from warnings import warn

import pandas as pd
from sklearn import clone
from sklearn.utils.metaestimators import if_delegate_has_method

from sktime.base import _HeterogenousMetaEstimator
from sktime.datatypes import ALL_TIME_SERIES_MTYPES
from sktime.transformations._delegate import _DelegatedTransformer
from sktime.transformations.base import BaseTransformer
from sktime.utils.multiindex import flatten_multiindex
from sktime.utils.sklearn import (
    is_sklearn_classifier,
    is_sklearn_clusterer,
    is_sklearn_regressor,
    is_sklearn_transformer,
)
from sktime.utils.validation.series import check_series

__author__ = ["fkiraly", "mloning", "miraep8", "aiwalter", "SveaMeyer13"]
__all__ = [
    "ColumnwiseTransformer",
    "FeatureUnion",
    "FitInTransform",
    "Id",
    "InvertTransform",
    "MultiplexTransformer",
    "OptionalPassthrough",
    "TransformerPipeline",
    "YtoX",
]


# mtypes for Series, Panel, Hierarchical,
# with exception of some ambiguous and discouraged mtypes
CORE_MTYPES = [
    "pd.DataFrame",
    "np.ndarray",
    "pd.Series",
    "pd-multiindex",
    "df-list",
    "nested_univ",
    "numpy3D",
    "pd_multiindex_hier",
]


def _coerce_to_sktime(other):
    """Check and format inputs to dunders for compose."""
    from sktime.transformations.series.adapt import TabularToSeriesAdaptor

    # if sklearn transformer, adapt to sktime transformer first
    if is_sklearn_transformer(other):
        return TabularToSeriesAdaptor(other)

    return other


class TransformerPipeline(_HeterogenousMetaEstimator, BaseTransformer):
    """Pipeline of transformers compositor.

    The `TransformerPipeline` compositor allows to chain transformers.
    The pipeline is constructed with a list of sktime transformers, i.e.
    estimators following the BaseTransformer interface. The list can be
    unnamed (a simple list of transformers) or string named (a list of
    pairs of string, estimator).

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN`,
    the pipeline behaves as follows:

    * `fit`
        Changes state by running `trafo1.fit_transform`,
        trafo2.fit_transform` etc sequentially, with
        `trafo[i]` receiving the output of `trafo[i-1]`
    * `transform`
        Result is of executing `trafo1.transform`, `trafo2.transform`,
        etc with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        and returning the output of `trafoN.transform`
    * `inverse_transform`
        Result is of executing `trafo[i].inverse_transform`,
        with `trafo[i].inverse_transform` input = output
        `trafo[i-1].inverse_transform`, and returning the output of
        `trafoN.inverse_transform`
    * `update`
        Changes state by chaining `trafo1.update`, `trafo1.transform`,
        `trafo2.update`, `trafo2.transform`, ..., `trafoN.update`,
        where `trafo[i].update` and `trafo[i].transform` receive as input
        the output of `trafo[i-1].transform`

    The `get_params`, `set_params` uses `sklearn` compatible nesting interface
    if list is unnamed, names are generated as names of classes
    if names are non-unique, `f"_{str(i)}"` is appended to each name string
    where `i` is the total count of occurrence of a non-unique string
    inside the list of names leading up to it (inclusive)

    A `TransformerPipeline` can also be created by using the magic multiplication
    on any transformer, i.e., any estimator inheriting from `BaseTransformer`
    for instance, `my_trafo1 * my_trafo2 * my_trafo3`
    will result in the same object as  obtained from the constructor
    `TransformerPipeline([my_trafo1, my_trafo2, my_trafo3])`
    A magic multiplication can also be used with (str, transformer) pairs,
    as long as one element in the chain is a transformer

    Parameters
    ----------
    steps : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when `fit` is called

    Attributes
    ----------
    steps_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in `steps` which are fitted in the pipeline
        is always in (str, transformer) format, even if `steps` is just a list
        strings not passed in `steps` are replaced by unique generated strings
        i-th transformer in `steps_` is clone of i-th in `steps`

    Examples
    --------
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> t1 = ExponentTransformer(power=2)
    >>> t2 = ExponentTransformer(power=0.5)

        Example 1, option A: construct without strings (unique names are generated for
        the two components t1 and t2)
    >>> pipe = TransformerPipeline(steps = [t1, t2])

        Example 1, option B: construct with strings to give custom names to steps
    >>> pipe = TransformerPipeline(
    ...         steps = [
    ...             ("trafo1", t1),
    ...             ("trafo2", t2),
    ...         ]
    ...     )

        Example 1, option C: for quick construction, the * dunder method can be used
    >>> pipe = t1 * t2

        Example 2: sklearn transformers can be used in the pipeline.
        If applied to Series, sklearn transformers are applied by series instance.
        If applied to Table, sklearn transformers are applied to the table as a whole.
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sktime.transformations.series.summarize import SummaryTransformer

        This applies the scaler per series, then summarizes:
    >>> pipe = StandardScaler() * SummaryTransformer()

        This applies the sumamrization, then scales the full summary table:
    >>> pipe = SummaryTransformer() * StandardScaler()

        This scales the series, then summarizes, then scales the full summary table:
    >>> pipe = StandardScaler() * SummaryTransformer() * StandardScaler()
    """

    _tags = {
        # we let all X inputs through to be handled by first transformer
        "X_inner_mtype": CORE_MTYPES,
        "univariate-only": False,
    }

    # no further default tag values - these are set dynamically below

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator) pairs for the default
    _steps_attr = "_steps"

    def __init__(self, steps):

        self.steps = steps
        self.steps_ = self._check_estimators(self.steps, cls_type=BaseTransformer)

        super(TransformerPipeline, self).__init__()

        # abbreviate for readability
        ests = self.steps_
        first_trafo = ests[0][1]
        last_trafo = ests[-1][1]

        # input mtype and input type are as of the first estimator
        self.clone_tags(first_trafo, ["scitype:transform-input"])
        # output type is that of last estimator, if no "Primitives" occur in the middle
        # if "Primitives" occur in the middle, then output is set to that too
        # this is in a case where "Series-to-Series" is applied to primitive df
        #   e.g., in a case of pipelining with scikit-learn transformers
        last_out = last_trafo.get_tag("scitype:transform-output")
        self._anytagis_then_set(
            "scitype:transform-output", "Primitives", last_out, ests
        )

        # set property tags based on tags of components
        self._anytag_notnone_set("y_inner_mtype", ests)
        self._anytag_notnone_set("scitype:transform-labels", ests)

        self._anytagis_then_set("scitype:instancewise", False, True, ests)
        self._anytagis_then_set("fit_is_empty", False, True, ests)
        self._anytagis_then_set("transform-returns-same-time-index", False, True, ests)
        self._anytagis_then_set("skip-inverse-transform", False, True, ests)

        # self can inverse transform if for all est, we either skip or can inv-trasform
        skips = [est.get_tag("skip-inverse-transform") for _, est in ests]
        has_invs = [est.get_tag("capability:inverse_transform") for _, est in ests]
        can_inv = [x or y for x, y in zip(skips, has_invs)]
        self.set_tags(**{"capability:inverse_transform": all(can_inv)})

        # can handle missing data iff all estimators can handle missing data
        #   up to a potential estimator when missing data is removed
        # removes missing data iff can handle missing data,
        #   and there is an estimator in the chain that removes it
        self._tagchain_is_linked_set(
            "handles-missing-data", "capability:missing_values:removes", ests
        )
        # can handle unequal length iff all estimators can handle unequal length
        #   up to a potential estimator which turns the series equal length
        # removes unequal length iff can handle unequal length,
        #   and there is an estimator in the chain that renders series equal length
        self._tagchain_is_linked_set(
            "capability:unequal_length", "capability:unequal_length:removes", ests
        )

    @property
    def _steps(self):
        return self._get_estimator_tuples(self.steps, clone_ests=False)

    @_steps.setter
    def _steps(self, value):
        self.steps = value

    def __mul__(self, other):
        """Magic * method, return (right) concatenated TransformerPipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of `self` (first) with `other` (last).
            not nested, contains only non-TransformerPipeline `sktime` transformers
        """
        from sktime.classification.compose import SklearnClassifierPipeline
        from sktime.clustering.compose import SklearnClustererPipeline
        from sktime.regression.compose import SklearnRegressorPipeline

        other = _coerce_to_sktime(other)

        # if sklearn classifier, use sklearn classifier pipeline
        if is_sklearn_classifier(other):
            return SklearnClassifierPipeline(classifier=other, transformers=self.steps)

        # if sklearn clusterer, use sklearn clusterer pipeline
        if is_sklearn_clusterer(other):
            return SklearnClustererPipeline(clusterer=other, transformers=self.steps)

        # if sklearn regressor, use sklearn regressor pipeline
        if is_sklearn_regressor(other):
            return SklearnRegressorPipeline(regressor=other, transformers=self.steps)

        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=TransformerPipeline,
            attr_name="steps",
            concat_order="left",
        )

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated TransformerPipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of `other` (first) with `self` (last).
            not nested, contains only non-TransformerPipeline `sktime` steps
        """
        other = _coerce_to_sktime(other)
        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=TransformerPipeline,
            attr_name="steps",
            concat_order="right",
        )

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: reference to self
        """
        self.steps_ = self._check_estimators(self.steps, cls_type=BaseTransformer)

        Xt = X
        for _, transformer in self.steps_:
            Xt = transformer.fit_transform(X=Xt, y=y)

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        Xt = X
        for _, transformer in self.steps_:
            if not self.get_tag("fit_is_empty", False):
                Xt = transformer.transform(X=Xt, y=y)
            else:
                Xt = transformer.fit_transform(X=Xt, y=y)

        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be inverse transformed
        y : Series or Panel of mtype y_inner_mtype, optional (default=None)
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        Xt = X
        for _, transformer in reversed(self.steps_):
            if not self.get_tag("fit_is_empty", False):
                Xt = transformer.inverse_transform(X=Xt, y=y)
            else:
                Xt = transformer.fit(X=Xt, y=y).inverse_transform(X=Xt, y=y)

        return Xt

    def _update(self, X, y=None):
        """Update transformer with X and y.

        private _update containing the core logic, called from update

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _update must support all types in it
            Data to update transformer with
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: reference to self
        """
        Xt = X
        for _, transformer in self.steps_:
            transformer.update(X=Xt, y=y)
            Xt = transformer.transform(X=Xt, y=y)

        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        # imports
        from sktime.transformations.series.exponent import ExponentTransformer

        t1 = ExponentTransformer(power=2)
        t2 = ExponentTransformer(power=0.5)
        t3 = ExponentTransformer(power=1)

        # construct without names
        params1 = {"steps": [t1, t2]}

        # construct with names
        params2 = {"steps": [("foo", t1), ("bar", t2), ("foobar", t3)]}

        # construct with names and provoke multiple naming clashes
        params3 = {"steps": [("foo", t1), ("foo", t2), ("foo_1", t3)]}

        return [params1, params2, params3]


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
        # unclear what inverse transform should be, since multiple inverse_transform
        #   would have to inverse transform to one
    }

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator) pairs for the default
    _steps_attr = "_transformer_list"

    def __init__(
        self,
        transformer_list,
        n_jobs=None,
        transformer_weights=None,
        flatten_transform_index=True,
    ):

        self.transformer_list = transformer_list
        self.transformer_list_ = self._check_estimators(
            transformer_list, cls_type=BaseTransformer
        )

        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.flatten_transform_index = flatten_transform_index

        super(FeatureUnion, self).__init__()

        # todo: check for transform-input, transform-output
        #   for now, we assume it's always Series/Series or Series/Panel
        #   but no error is currently raised

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

    @property
    def _transformer_list(self):
        return self._get_estimator_tuples(self.transformer_list, clone_ests=False)

    @_transformer_list.setter
    def _transformer_list(self, value):
        self.transformer_list = value
        self.transformer_list_ = self._check_estimators(value, cls_type=BaseTransformer)

    def __add__(self, other):
        """Magic + method, return (right) concatenated FeatureUnion.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of `self` (first) with `other` (last).
            not nested, contains only non-FeatureUnion `sktime` transformers
        """
        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=FeatureUnion,
            attr_name="transformer_list",
            concat_order="left",
        )

    def __radd__(self, other):
        """Magic + method, return (left) concatenated FeatureUnion.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of `self` (last) with `other` (first).
            not nested, contains only non-FeatureUnion `sktime` transformers
        """
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


class FitInTransform(BaseTransformer):
    """Transformer wrapper to delay fit to the transform phase.

    In panel settings, e.g., time series classification, it can be preferable
    (or, necessary) to fit and transform on the test set, e.g., interpolate within the
    same series that interpolation parameters are being fitted on. `FitInTransform` can
    be used to wrap any transformer to ensure that `fit` and `transform` happen always
    on the same series, by delaying the `fit` to the `transform` batch.

    Warning: The use of `FitInTransform` will typically not be useful, or can constitute
    a mistake (data leakage) when naively used in a forecasting setting.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series.
    skip_inverse_transform : bool
        The FitInTransform will skip inverse_transform by default, of the param
        skip_inverse_transform=False, then the inverse_transform is calculated
        by means of transformer.fit(X=X, y=y).inverse_transform(X=X, y=y) where
        transformer is the inner transformer. So the inner transformer is
        fitted on the inverse_transform data. This is required to have a non-
        state changing transform() method of FitInTransform.

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> from sktime.transformations.compose import FitInTransform
    >>> from sktime.transformations.series.impute import Imputer
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> # we want to fit the Imputer only on the predict (=transform) data.
    >>> # note that NaiveForecaster cant use X data, this is just a show case.
    >>> pipe = ForecastingPipeline(
    ...     steps=[
    ...         ("imputer", FitInTransform(Imputer(method="mean"))),
    ...         ("forecaster", NaiveForecaster()),
    ...     ]
    ... )
    >>> pipe.fit(y_train, X_train)
    ForecastingPipeline(...)
    >>> y_pred = pipe.predict(fh=fh, X=X_test)
    """

    def __init__(self, transformer, skip_inverse_transform=True):
        self.transformer = transformer
        self.skip_inverse_transform = skip_inverse_transform
        super(FitInTransform, self).__init__()
        self.clone_tags(transformer, None)
        self.set_tags(
            **{
                "fit_is_empty": True,
                "skip-inverse-transform": self.skip_inverse_transform,
            }
        )

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        return clone(self.transformer).fit_transform(X=X, y=y)

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be inverse transformed
        y : Series or Panel of mtype y_inner_mtype, optional (default=None)
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        return clone(self.transformer).fit(X=X, y=y).inverse_transform(X=X, y=y)

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.transformations.series.boxcox import BoxCoxTransformer

        params = [
            {"transformer": BoxCoxTransformer()},
            {"transformer": BoxCoxTransformer(), "skip_inverse_transform": False},
        ]
        return params


class MultiplexTransformer(_HeterogenousMetaEstimator, _DelegatedTransformer):
    """Facilitate an AutoML based selection of the best transformer.

    When used in combination with either TransformedTargetForecaster or
    ForecastingPipeline in combination with ForecastingGridSearchCV
    MultiplexTransformer provides a framework for transformer selection.  Through
    selection of the appropriate pipeline (ie TransformedTargetForecaster vs
    ForecastingPipeline) the transformers in MultiplexTransformer will either be
    applied to exogenous data, or to the target data.

    MultiplexTransformer delegates all transforming tasks (ie, calls to fit, transform,
    inverse_transform, and update) to a copy of the transformer in transformers
    whose name matches selected_transformer.  All other transformers in transformers
    will be ignored.

    Parameters
    ----------
    transformers : list of sktime transformers, or
        list of tuples (str, estimator) of named sktime transformers
        MultiplexTransformer can switch ("multiplex") between these transformers.
        Note - all the transformers passed in "transformers" should be thought of as
        blueprints.  Calling transformation functions on MultiplexTransformer will not
        change their state at all. - Rather a copy of each is created and this is what
        is updated.
    selected_transformer: str or None, optional, Default=None.
        If str, must be one of the transformer names.
            If passed in transformers were unnamed then selected_transformer must
            coincide with auto-generated name strings.
            To inspect auto-generated name strings, call get_params.
        If None, selected_transformer defaults to the name of the first transformer
           in transformers.
        selected_transformer represents the name of the transformer MultiplexTransformer
           should behave as (ie delegate all relevant transformation functionality to)

    Attributes
    ----------
    transformer_ : sktime transformer
        clone of the transformer named by selected_transformer to which all the
        transformation functionality is delegated to.
    _transformers : list of (name, est) tuples, where est are direct references to
        the estimators passed in transformers passed. If transformers was passed
        without names, those be auto-generated and put here.

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.compose import MultiplexTransformer
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.model_selection import (
    ...     ForecastingGridSearchCV,
    ...     ExpandingWindowSplitter)
    >>> # create MultiplexTransformer:
    >>> multiplexer = MultiplexTransformer(transformers=[
    ...     ("impute_mean", Imputer(method="mean", missing_values = -1)),
    ...     ("impute_near", Imputer(method="nearest", missing_values = -1)),
    ...     ("impute_rand", Imputer(method="random", missing_values = -1))])
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=24,
    ...     step_length=12,
    ...     fh=[1,2,3])
    >>> pipe = TransformedTargetForecaster(steps = [
    ...     ("multiplex", multiplexer),
    ...     ("forecaster", NaiveForecaster())
    ...     ])
    >>> gscv = ForecastingGridSearchCV(
    ...     cv=cv,
    ...     param_grid={"multiplex__selected_transformer":
    ...     ["impute_mean", "impute_near", "impute_rand"]},
    ...     forecaster=pipe,
    ...     )
    >>> y = load_shampoo_sales()
    >>> # randomly make some of the values nans:
    >>> y.loc[y.sample(frac=0.1).index] = -1
    >>> gscv = gscv.fit(y)
    """

    # tags will largely be copied from selected_transformer
    _tags = {
        "fit_is_empty": False,
        "univariate-only": False,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
    }

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator) pairs for the default
    _steps_attr = "_transformers"

    def __init__(
        self,
        transformers: list,
        selected_transformer=None,
    ):
        super(MultiplexTransformer, self).__init__()
        self.selected_transformer = selected_transformer

        self.transformers = transformers
        self._check_estimators(
            transformers,
            attr_name="transformers",
            cls_type=BaseTransformer,
            clone_ests=False,
        )
        self._set_transformer()
        self.clone_tags(self.transformer_)
        self.set_tags(**{"fit_is_empty": False})
        # this ensures that we convert in the inner estimator, not in the multiplexer
        self.set_tags(**{"X_inner_mtype": ALL_TIME_SERIES_MTYPES})

    @property
    def _transformers(self):
        """Forecasters turned into name/est tuples."""
        return self._get_estimator_tuples(self.transformers, clone_ests=False)

    @_transformers.setter
    def _transformers(self, value):
        self.transformers = value

    def _check_selected_transformer(self):
        component_names = self._get_estimator_names(
            self._transformers, make_unique=True
        )
        selected = self.selected_transformer
        if selected is not None and selected not in component_names:
            raise Exception(
                f"Invalid selected_transformer parameter value provided, "
                f" found: {selected}. Must be one of these"
                f" valid selected_transformer parameter values: {component_names}."
            )

    def _set_transformer(self):
        self._check_selected_transformer()
        # clone the selected transformer to self.transformer_
        if self.selected_transformer is not None:
            for name, transformer in self._get_estimator_tuples(self.transformers):
                if self.selected_transformer == name:
                    self.transformer_ = transformer.clone()
        else:
            # if None, simply clone the first transformer to self.transformer_
            self.transformer_ = self._get_estimator_list(self.transformers)[0].clone()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.transformations.series.impute import Imputer

        # test with 2 simple detrend transformations with selected_transformer
        params1 = {
            "transformers": [
                ("imputer_mean", Imputer(method="mean")),
                ("imputer_near", Imputer(method="nearest")),
            ],
            "selected_transformer": "imputer_near",
        }
        # test no selected_transformer
        params2 = {
            "transformers": [
                Imputer(method="mean"),
                Imputer(method="nearest"),
            ],
        }
        return [params1, params2]

    def __or__(self, other):
        """Magic | (or) method, return (right) concatenated MultiplexTransformer.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        MultiplexTransformer object, concatenation of `self` (first) with `other`
            (last).not nested, contains only non-MultiplexTransformer `sktime`
            transformers

        Raises
        ------
        ValueError if other is not of type MultiplexTransformer or BaseTransformer.
        """
        other = _coerce_to_sktime(other)
        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=MultiplexTransformer,
            attr_name="transformers",
            concat_order="left",
        )

    def __ror__(self, other):
        """Magic | (or) method, return (left) concatenated MultiplexTransformer.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        MultiplexTransformer object, concatenation of `self` (last) with `other`
            (first). not nested, contains only non-MultiplexTransformer `sktime`
            transformers
        """
        other = _coerce_to_sktime(other)
        return self._dunder_concat(
            other=other,
            base_class=BaseTransformer,
            composite_class=MultiplexTransformer,
            attr_name="forecasters",
            concat_order="right",
        )


class InvertTransform(_DelegatedTransformer):
    """Invert a series-to-series transformation.

    Switches `transform` and `inverse_transform`, leaves `fit` and `update` the same.

    Parameters
    ----------
    transformer : sktime transformer, must transform Series input to Series output
        this is a "blueprint" transformer, state does not change when `fit` is called

    Attributes
    ----------
    transformer_: transformer,
        this clone is fitted when `fit` is called and provides `transform` and inverse

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.compose import InvertTransform
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>>
    >>> inverse_exponent = InvertTransform(ExponentTransformer(power=3))
    >>> X = load_airline()
    >>> Xt = inverse_exponent.fit_transform(X)  # computes 3rd square root
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": False,
        "capability:inverse_transform": True,
    }

    def __init__(self, transformer):
        self.transformer = transformer

        super(InvertTransform, self).__init__()

        self.transformer_ = transformer.clone()

        # should be all tags, but not fit_is_empty
        #   (_fit should not be skipped)
        tags_to_clone = [
            "scitype:transform-input",
            "scitype:transform-output",
            "scitype:instancewise",
            "X_inner_mtype",
            "y_inner_mtype",
            "handles-missing-data",
            "X-y-must-have-same-index",
            "transform-returns-same-time-index",
            "skip-inverse-transform",
        ]
        self.clone_tags(transformer, tag_names=tags_to_clone)

        if not transformer.get_tag("capability:inverse_transform", False):
            warn(
                "transformer does not have capability to inverse transform, "
                "according to capability:inverse_transform tag. "
                "If the tag was correctly set, this transformer will likely crash"
            )
        inner_output = transformer.get_tag("scitype:transform-output")
        if transformer.get_tag("scitype:transform-output") != "Series":
            warn(
                f"transformer output is not Series but {inner_output}, "
                "according to scitype:transform-output tag. "
                "The InvertTransform wrapper supports only Series output, therefore"
                " this transformer will likely crash on input."
            )

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Returns a transformed version of X by iterating over specified
        columns and applying the wrapped transformer to them.

        Parameters
        ----------
        X : sktime compatible time series container
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : sktime compatible time series container
            transformed version of X
        """
        return self.transformer_.inverse_transform(X=X, y=y)

    def _inverse_transform(self, X, y=None):
        """Logic used by `inverse_transform` to reverse transformation on `X`.

        Returns an inverse-transformed version of X by iterating over specified
        columns and applying the univariate series transformer to them.

        Only works if `self.transformer` has an `inverse_transform` method.

        Parameters
        ----------
        X : sktime compatible time series container
            Data to be inverse transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : sktime compatible time series container
            inverse transformed version of X
        """
        return self.transformer_.transform(X=X, y=y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.transformations.series.boxcox import BoxCoxTransformer
        from sktime.transformations.series.exponent import ExponentTransformer

        # ExponentTransformer skips fit
        params1 = {"transformer": ExponentTransformer()}
        # BoxCoxTransformer has fit
        params2 = {"transformer": BoxCoxTransformer()}

        return [params1, params2]


class Id(BaseTransformer):
    """Identity transformer, returns data unchanged in transform/inverse_transform."""

    _tags = {
        "capability:inverse_transform": True,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": CORE_MTYPES,  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
        "handles-missing-data": True,  # can estimator handle missing data?
    }

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : any sktime compatible data, Series, Panel, or Hierarchical
        y : optional, default=None
            ignored, argument present for interface conformance

        Returns
        -------
        X, identical to input
        """
        return X

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : any sktime compatible data, Series, Panel, or Hierarchical
        y : optional, default=None
            ignored, argument present for interface conformance

        Returns
        -------
        X, identical to input
        """
        return X

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        return {}


class OptionalPassthrough(_DelegatedTransformer):
    """Wrap an existing transformer to tune whether to include it in a pipeline.

    Allows tuning the implicit hyperparameter whether or not to use a
    particular transformer inside a pipeline (e.g. TransformedTargetForecaster)
    or not. This is achieved by the hyperparameter `passthrough`
    which can be added to a tuning grid then (see example).

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series.
        this is a "blueprint" transformer, state does not change when `fit` is called
    passthrough : bool, default=False
       Whether to apply the given transformer or to just
        passthrough the data (identity transformation). If, True the transformer
        is not applied and the OptionalPassthrough uses the identity
        transformation.

    Attributes
    ----------
    transformer_: transformer,
        this clone is fitted when `fit` is called and provides `transform` and inverse
        if passthrough = False, a clone of `transformer`passed
        if passthrough = True, the identity transformer `Id`

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.compose import OptionalPassthrough
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.model_selection import (
    ...     ForecastingGridSearchCV,
    ...     SlidingWindowSplitter)
    >>> from sklearn.preprocessing import StandardScaler
    >>> # create pipeline
    >>> pipe = TransformedTargetForecaster(steps=[
    ...     ("deseasonalizer", OptionalPassthrough(Deseasonalizer())),
    ...     ("scaler", OptionalPassthrough(TabularToSeriesAdaptor(StandardScaler()))),
    ...     ("forecaster", NaiveForecaster())])  # doctest: +SKIP
    >>> # putting it all together in a grid search
    >>> cv = SlidingWindowSplitter(
    ...     initial_window=60,
    ...     window_length=24,
    ...     start_with_window=True,
    ...     step_length=48)  # doctest: +SKIP
    >>> param_grid = {
    ...     "deseasonalizer__passthrough" : [True, False],
    ...     "scaler__transformer__transformer__with_mean": [True, False],
    ...     "scaler__passthrough" : [True, False],
    ...     "forecaster__strategy": ["drift", "mean", "last"]}  # doctest: +SKIP
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=pipe,
    ...     param_grid=param_grid,
    ...     cv=cv,
    ...     n_jobs=-1)  # doctest: +SKIP
    >>> gscv_fitted = gscv.fit(load_airline())  # doctest: +SKIP
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": CORE_MTYPES,
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": False,
        "capability:inverse_transform": True,
    }

    def __init__(self, transformer, passthrough=False):
        self.transformer = transformer
        self.passthrough = passthrough

        super(OptionalPassthrough, self).__init__()

        # should be all tags, but not fit_is_empty
        #   (_fit should not be skipped)
        tags_to_clone = [
            "scitype:transform-input",
            "scitype:transform-output",
            "scitype:instancewise",
            "X_inner_mtype",
            "y_inner_mtype",
            "capability:inverse_transform",
            "handles-missing-data",
            "X-y-must-have-same-index",
            "transform-returns-same-time-index",
            "skip-inverse-transform",
        ]
        self.clone_tags(transformer, tag_names=tags_to_clone)

        if passthrough:
            self.transformer_ = Id()
        else:
            self.transformer_ = transformer.clone()

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.transformations.series.boxcox import BoxCoxTransformer

        return {"transformer": BoxCoxTransformer(), "passthrough": False}


class ColumnwiseTransformer(BaseTransformer):
    """Apply a transformer columnwise to multivariate series.

    Overview: input multivariate time series and the transformer passed
    in `transformer` parameter is applied to specified `columns`, each
    column is handled as a univariate series. The resulting transformed
    data has the same shape as input data.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series.
    columns : list of str or None
            Names of columns that are supposed to be transformed.
            If None, all columns are transformed.

    Attributes
    ----------
    transformers_ : dict of {str : transformer}
        Maps columns to transformers.
    columns_ : list of str
        Names of columns that are supposed to be transformed.

    See Also
    --------
    OptionalPassthrough

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.transformations.series.detrend import Detrender
    >>> from sktime.transformations.compose import ColumnwiseTransformer
    >>> _, X = load_longley()
    >>> transformer = ColumnwiseTransformer(Detrender())
    >>> Xt = transformer.fit_transform(X)
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": False,
    }

    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns
        super(ColumnwiseTransformer, self).__init__()

        tags_to_clone = [
            "y_inner_mtype",
            "capability:inverse_transform",
            "handles-missing-data",
            "X-y-must-have-same-index",
            "transform-returns-same-time-index",
            "skip-inverse-transform",
        ]
        self.clone_tags(transformer, tag_names=tags_to_clone)

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit transform to
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        # check that columns are None or list of strings
        if self.columns is not None:
            if not isinstance(self.columns, list) and all(
                isinstance(s, str) for s in self.columns
            ):
                raise ValueError("Columns need to be a list of strings or None.")

        # set self.columns_ to columns that are going to be transformed
        # (all if self.columns is None)
        self.columns_ = self.columns
        if self.columns_ is None:
            self.columns_ = X.columns

        # make sure z contains all columns that the user wants to transform
        _check_columns(X, selected_columns=self.columns_)

        # fit by iterating over columns
        self.transformers_ = {}
        for colname in self.columns_:
            transformer = self.transformer.clone()
            self.transformers_[colname] = transformer
            self.transformers_[colname].fit(X[colname], y)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Returns a transformed version of X by iterating over specified
        columns and applying the wrapped transformer to them.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.DataFrame
            transformed version of X
        """
        # make copy of z
        X = X.copy()

        # make sure z contains all columns that the user wants to transform
        _check_columns(X, selected_columns=self.columns_)
        for colname in self.columns_:
            X[colname] = self.transformers_[colname].transform(X[colname], y)
        return X

    def _inverse_transform(self, X, y=None):
        """Logic used by `inverse_transform` to reverse transformation on `X`.

        Returns an inverse-transformed version of X by iterating over specified
        columns and applying the univariate series transformer to them.
        Only works if `self.transformer` has an `inverse_transform` method.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be inverse transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.DataFrame
            inverse transformed version of X
        """
        # make copy of z
        X = X.copy()

        # make sure z contains all columns that the user wants to transform
        _check_columns(X, selected_columns=self.columns_)

        # iterate over columns that are supposed to be inverse_transformed
        for colname in self.columns_:
            X[colname] = self.transformers_[colname].inverse_transform(X[colname], y)

        return X

    @if_delegate_has_method(delegate="transformer")
    def update(self, X, y=None, update_params=True):
        """Update parameters.

        Update the parameters of the estimator with new data
        by iterating over specified columns.
        Only works if `self.transformer` has an `update` method.

        Parameters
        ----------
        X : pd.Series
            New time series.
        update_params : bool, optional, default=True

        Returns
        -------
        self : an instance of self
        """
        z = check_series(X)

        # make z a pd.DataFrame in univariate case
        if isinstance(z, pd.Series):
            z = z.to_frame()

        # make sure z contains all columns that the user wants to transform
        _check_columns(z, selected_columns=self.columns_)
        for colname in self.columns_:
            self.transformers_[colname].update(z[colname], X)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.transformations.series.detrend import Detrender

        return {"transformer": Detrender()}


def _check_columns(z, selected_columns):
    # make sure z contains all columns that the user wants to transform
    z_wanted_keys = set(selected_columns)
    z_new_keys = set(z.columns)
    difference = z_wanted_keys.difference(z_new_keys)
    if len(difference) != 0:
        raise ValueError("Missing columns" + str(difference) + "in Z.")


def _check_is_pdseries(z):
    # make z a pd.Dataframe in univariate case
    is_series = False
    if isinstance(z, pd.Series):
        z = z.to_frame()
        is_series = True
    return z, is_series


class YtoX(BaseTransformer):
    """Create exogeneous features which are a copy of the endogenous data.

    Replaces exogeneous features (`X`) by endogeneous data (`y`).

    To *add* instead of *replace*, use `FeatureUnion`.

    Parameters
    ----------
    subset_index : boolean, optional, default=False
        if True, subsets the output of `transform` to `X.index`,
        i.e., outputs `y.loc[X.index]`
    """

    _tags = {
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": False,
        "univariate-only": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "both",
        "fit_is_empty": True,
        "requires_y": True,
    }

    def __init__(self, subset_index=False):

        self.subset_index = subset_index

        super(YtoX, self).__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : time series or panel in one of the pd.DataFrame formats
            Data to be transformed
        y : time series or panel in one of the pd.DataFrame formats
            Additional data, e.g., labels for transformation

        Returns
        -------
        y, as a transformed version of X
        """
        if self.subset_index:
            return y.loc[X.index.intersection(y.index)]
        else:
            return y

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        Drops featurized column that was added in transform().

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be inverse transformed
        y : Series or Panel of mtype y_inner_mtype, optional (default=None)
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        if self.subset_index:
            return y.loc[X.index.intersection(y.index)]
        else:
            return y
