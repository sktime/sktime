# -*- coding: utf-8 -*-
"""Meta-transformers for building composite transformers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd
from sklearn import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations._delegate import _DelegatedTransformer
from sktime.transformations.base import BaseTransformer
from sktime.utils.multiindex import flatten_multiindex
from sktime.utils.sklearn import is_sklearn_classifier, is_sklearn_transformer

__author__ = ["fkiraly", "mloning", "miraep8"]
__all__ = [
    "TransformerPipeline",
    "FeatureUnion",
    "FitInTransform",
    "MultiplexTransformer",
]


def _coerce_to_sktime(other):
    """Check and format inputs to dunders for compose."""
    from sktime.transformations.series.adapt import TabularToSeriesAdaptor

    # if sklearn transformer, adapt to sktime transformer first
    if is_sklearn_transformer(other):
        return TabularToSeriesAdaptor(other)

    return other


class TransformerPipeline(BaseTransformer, _HeterogenousMetaEstimator):
    """Pipeline of transformers compositor.

    The `TransformerPipeline` compositor allows to chain transformers.
    The pipeline is constructed with a list of sktime transformers,
        i.e., estimators following the BaseTransformer interface.
    The list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN`,
        the pipeline behaves as follows:
    `fit` - changes state by running `trafo1.fit_transform`, `trafo2.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`
    `transform` - result is of executing `trafo1.transform`, `trafo2.transform`, etc
        with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        and returning the output of `trafoN.transform`
    `inverse_transform` - result is of executing `trafo[i].inverse_transform`,
        with `trafo[i].inverse_transform` input = output `trafo[i-1].inverse_transform`,
        and returning the output of `trafoN.inverse_transform`
    `update` - changes state by chaining `trafo1.update`, `trafo1.transform`,
        `trafo2.update`, `trafo2.transform`, ..., `trafoN.update`,
        where `trafo[i].update` and `trafo[i].transform` receive as input
            the output of `trafo[i-1].transform`

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `TransformerPipeline` can also be created by using the magic multiplication
        on any transformer, i.e., any estimator inheriting from `BaseTransformer`
            for instance, `my_trafo1 * my_trafo2 * my_trafo3`
            will result in the same object as  obtained from the constructor
            `TransformerPipeline([my_trafo1, my_trafo2, my_trafo3])`
        magic multiplication can also be used with (str, transformer) pairs,
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
    We'll construct a pipeline from 2 transformers below, in three different ways
    Preparing the transformers
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> t1 = ExponentTransformer(power=2)
    >>> t2 = ExponentTransformer(power=0.5)

    Example 1, option A: construct without strings
    >>> pipe = TransformerPipeline(steps = [t1, t2])
    >>> # unique names are generated for the two components t1 and t2

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

    _required_parameters = ["steps"]

    _tags = {
        # we let all X inputs through to be handled by first transformer
        "X_inner_mtype": [
            "pd.DataFrame",
            "np.ndarray",
            "pd.Series",
            "pd-multiindex",
            "df-list",
            "nested_univ",
            "numpy3D",
            "pd_multiindex_hier",
        ],
        "univariate-only": False,
    }

    # no further default tag values - these are set dynamically below

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
        self._anytagis_then_set("capability:inverse_transform", False, True, ests)

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

        other = _coerce_to_sktime(other)

        # if sklearn classifier, use sklearn classifier pipeline
        if is_sklearn_classifier(other):
            return SklearnClassifierPipeline(classifier=other, transformers=self.steps)

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

    def get_params(self, deep=True):
        """Get parameters of estimator in `steps`.

        Parameters
        ----------
        deep : boolean, optional, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("_steps", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `steps`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params("_steps", **kwargs)
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
    flatten_transform_index : bool, optional (default=True)
        if True, columns of return DataFrame are flat, by "transformer__variablename"
        if False, columns are MultiIndex (transformer, variablename)
        has no effect if return mtype is one without column names
    """

    _required_parameters = ["transformer_list"]

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

    def get_params(self, deep=True):
        """Get parameters of estimator in `_forecasters`.

        Parameters
        ----------
        deep : boolean, optional, default=True
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
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for FeatureUnion."""
        from sktime.transformations.series.exponent import ExponentTransformer

        TRANSFORMERS = [
            ("transformer1", ExponentTransformer(power=4)),
            ("transformer2", ExponentTransformer(power=0.25)),
        ]

        return {"transformer_list": TRANSFORMERS}


class FitInTransform(BaseTransformer):
    """Transformer composition to always fit a given transformer on the transform data only.

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

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return self.transformer_.get_fitted_params()

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


class MultiplexTransformer(_DelegatedTransformer, _HeterogenousMetaEstimator):
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
    ...     start_with_window=True,
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
    }

    _delegate_name = "transformer_"

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

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("_transformers", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params("_transformers", **kwargs)
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
