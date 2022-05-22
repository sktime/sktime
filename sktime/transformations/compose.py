# -*- coding: utf-8 -*-
"""Meta-transformers for building composite transformers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from warnings import warn

import pandas as pd
from sklearn import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations.base import BaseTransformer

__author__ = ["fkiraly", "mloning"]
__all__ = ["TransformerPipeline", "FeatureUnion"]


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
    >>> # we'll construct a pipeline from 2 transformers below, in three different ways
    >>> # preparing the transformers
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>> t1 = ExponentTransformer(power=2)
    >>> t2 = ExponentTransformer(power=0.5)

    >>> # Example 1: construct without strings
    >>> pipe = TransformerPipeline(steps = [t1, t2])
    >>> # unique names are generated for the two components t1 and t2

    >>> # Example 2: construct with strings to give custom names to steps
    >>> pipe = TransformerPipeline(
    ...         steps = [
    ...             ("trafo1", t1),
    ...             ("trafo2", t2),
    ...         ]
    ...     )

    >>> # Example 3: for quick construction, the * dunder method can be used
    >>> pipe = t1 * t2
    """

    _required_parameters = ["steps"]

    # no default tag values - these are set dynamically below

    def __init__(self, steps):

        self.steps = steps
        self.steps_ = self._check_estimators(self.steps, cls_type=BaseTransformer)

        super(TransformerPipeline, self).__init__()

        first_trafo = self.steps_[0][1]
        last_trafo = self.steps_[-1][1]

        self.clone_tags(first_trafo, ["X_inner_mtype", "scitype:transform-input"])
        self.clone_tags(last_trafo, "scitype:transform-output")

        # abbreviate for readability
        ests = self.steps_

        # set property tags based on tags of components
        self._anytag_notnone_set("y_inner_mtype", ests)
        self._anytag_notnone_set("scitype:transform-labels", ests)

        self._anytagis_then_set("scitype:instancewise", False, True, ests)
        self._anytagis_then_set("X-y-must-have-same-index", True, False, ests)
        self._anytagis_then_set("fit_is_empty", False, True, ests)
        self._anytagis_then_set("transform-returns-same-time-index", False, True, ests)
        self._anytagis_then_set("skip-inverse-transform", True, False, ests)
        self._anytagis_then_set("capability:inverse_transform", False, True, ests)
        self._anytagis_then_set("univariate-only", True, False, ests)

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
        # need to escape if other is BaseForecaster
        #   this is because forecsting Pipelines are *also* transformers
        #   but they need to take precedence in parsing the expression
        from sktime.forecasting.base import BaseForecaster

        if isinstance(other, BaseForecaster):
            return NotImplemented

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
        # need to escape if other is BaseForecaster
        #   this is because forecsting Pipelines are *also* transformers
        #   but they need to take precedence in parsing the expression
        from sktime.forecasting.base import BaseForecaster

        if isinstance(other, BaseForecaster):
            return NotImplemented

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
        preserve_dataframe=True,
        flatten_transform_index=True,
    ):

        self.transformer_list = transformer_list
        self.transformer_list_ = self._check_estimators(
            transformer_list, cls_type=BaseTransformer
        )

        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.preserve_dataframe = preserve_dataframe
        if not preserve_dataframe:
            warn(
                "the preserve_dataframe arg has been deprecated in 0.11.0, "
                "and will be removed in 0.12.0. It has no effect on the output format, "
                "but can still be set to avoid compatibility issues in the deprecation "
                "period. FeatureUnion now follows the "
                "output format specification for sktime transformers. "
                "To convert the output to another format, use datatypes.convert_to"
            )
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
            flat_index = pd.Index([self._underscore_join(x) for x in Xt.columns])
            Xt.columns = flat_index

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

    @staticmethod
    def _underscore_join(iterable):
        """Create flattened column names from multiindex tuple."""
        iterable_as_str = [str(x) for x in iterable]
        return "__".join(iterable_as_str)


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


class ForecasterTransform(BaseTransformer):
    """Transformer that uses a forecaster to interpolate or extrapolate.

    Wraps a forecaster and a forecasting horizon specification, but acts as transformer.

    In `transform`, the wrapped forecaster carries out a `predict` step.
    The result from that prediction is added to the input of `transform`.
    In case of index clashes, the prediction overwrites the input.

    The typical use case is extending exogeneous data available only up until the cutoff
    into the future, for use by an exogeneous forecaster that requires such future data.

    Parameters
    ----------
    forecaster : BaseForecaster
        sktime forecaster to use for transformation
    fh : None, ForecastingHorizon, or valid input to construct ForecastingHorizon
        optional, default = None = in-sample, fh are indices of series seen in transform
        valid inputs to construct ForecastingHorizon are:
        int, list of int, 1D np.ndarray, pandas.Index (see ForecastingHorizon)
    behaviour : str, one of "update" or "refit", optional, default = "refit"
        if "refit", `fit` is empty, and forecaster is fit to data in `transform` only,
            Forecast added to `X` in `transform` is obtained from this state.
        if "update", forecaster is fit to the data batch seen in `fit`, then updated
            with any data seen in applications of `transform`.
            Forecast added to `X` in `transform` is obtained from this state.

    Attributes
    ----------
    forecaster_ : BaseForecaster
        clone of forecaster, only created if behaviour = "update".
        state is after being fitted to data in `fit` or updated with data in `transform`

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.arima import ARIMA
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> from sktime.forecasting.var import VAR
    >>> from sktime.transformations.compose import ForecasterTransform

    >>> y, X = load_longley()
    >>> fh = ForecastingHorizon([1, 2, 3])
    >>> pipe = ForecastingPipeline(
    ...     steps=[
    ...         ("exogeneous-forecast", ForecasterTransform(VAR(), fh=fh)),
    ...         ("forecaster", ARIMA()),
    ...     ]
    ... )
    >>> pipe.fit(y, X)
    ForecastingPipeline(...)
    >>> # this works without X from the future of y
    >>> y_pred = pipe.predict(fh=fh)
    """

    _tags = {
        "skip-inverse-transform": True,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
    }

    def __init__(self, forecaster, fh=None, behaviour="refit"):

        if behaviour not in ["update", "refit"]:
            raise ValueError('behaviour must be one of "update", "refit"')

        self.forecaster = forecaster
        self.fh = fh
        self.behaviour = behaviour
        super(ForecasterTransform, self).__init__()

        tag_translate_dict = {
            "handles-missing-data": forecaster.get_tag("handles-missing-data")
        }
        self.set_tags(**tag_translate_dict)

        if behaviour == "refit":
            self.set_tags(**{"fit_is_empty": True})

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit transform to
        y : ignored argument, present for interface compatibility

        Returns
        -------
        self: a fitted instance of the estimator
        """
        if self.behaviour == "update":
            self.forecaster_ = self.forecaster.clone()
            self.forecaster_.fit(y=X, fh=self.fh)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed
        y : ignored argument, present for interface compatibility

        Returns
        -------
        transformed version of X
        """
        if self.fh is None:
            fh = X.index
        else:
            fh = self.fh

        if self.behaviour == "refit":
            forecaster_ = self.forecaster.clone()
            forecaster_.fit(y=X, fh=fh)
            X_pred = forecaster_.predict()
        elif self.behavivour == "update":
            self.forecaster_.update(y=X)
            X_pred = forecaster_.predict()

        # return X_pred
        return X_pred.combine_first(X)

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
