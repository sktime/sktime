"""Transformer pipeline."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["TransformerPipeline"]

from sktime.base._meta import _HeterogenousMetaEstimator
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose._common import CORE_MTYPES, _coerce_to_sktime
from sktime.utils.sklearn import (
    is_sklearn_classifier,
    is_sklearn_clusterer,
    is_sklearn_regressor,
)


class TransformerPipeline(_HeterogenousMetaEstimator, BaseTransformer):
    """Pipeline of transformers compositor.

    The ``TransformerPipeline`` compositor allows to chain transformers.
    The pipeline is constructed with a list of sktime transformers, i.e.
    estimators following the BaseTransformer interface. The list can be
    unnamed (a simple list of transformers) or string named (a list of
    pairs of string, estimator).

    For a list of transformers ``trafo1``, ``trafo2``, ..., ``trafoN``,
    the pipeline behaves as follows:

    * ``fit``
        Changes state by running ``trafo1.fit_transform``,
        trafo2.fit_transform` etc sequentially, with
        ``trafo[i]`` receiving the output of ``trafo[i-1]``
    * ``transform``
        Result is of executing ``trafo1.transform``, ``trafo2.transform``,
        etc with ``trafo[i].transform`` input = output of ``trafo[i-1].transform``,
        and returning the output of ``trafoN.transform``
    * ``inverse_transform``
        Result is of executing ``trafo[i].inverse_transform``,
        with ``trafo[i].inverse_transform`` input = output
        ``trafo[i-1].inverse_transform``, and returning the output of
        ``trafoN.inverse_transform``
    * ``update``
        Changes state by chaining ``trafo1.update``, ``trafo1.transform``,
        ``trafo2.update``, ``trafo2.transform``, ..., ``trafoN.update``,
        where ``trafo[i].update`` and ``trafo[i].transform`` receive as input
        the output of ``trafo[i-1].transform``

    For transformers in the pipeline that use the ``y`` argument, the ``y`` argument
    passed to ``TransformerPipeline.fit`` or ``transform`` is passed to
    all such transformers in the pipeline. No transformations or inverse transformations
    are applied to ``y`` in the pipeline.

    The ``get_params``, ``set_params`` uses ``sklearn`` compatible nesting interface
    if list is unnamed, names are generated as names of classes
    if names are non-unique, ``f"_{str(i)}"`` is appended to each name string
    where ``i`` is the total count of occurrence of a non-unique string
    inside the list of names leading up to it (inclusive)

    A ``TransformerPipeline`` can also be created by using the magic multiplication
    on any transformer, i.e., any estimator inheriting from ``BaseTransformer``
    for instance, ``my_trafo1 * my_trafo2 * my_trafo3``
    will result in the same object as  obtained from the constructor
    ``TransformerPipeline([my_trafo1, my_trafo2, my_trafo3])``
    A magic multiplication can also be used with (str, transformer) pairs,
    as long as one element in the chain is a transformer

    Parameters
    ----------
    steps : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
        these are "blueprint" transformers, states do not change when ``fit`` is called

    Attributes
    ----------
    steps_ : list of tuples (str, transformer) of sktime transformers
        clones of transformers in ``steps`` which are fitted in the pipeline
        is always in (str, transformer) format, even if ``steps`` is just a list
        strings not passed in ``steps`` are replaced by unique generated strings
        i-th transformer in ``steps_`` is clone of i-th in ``steps``

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
        "authors": "fkiraly",
        # we let all X inputs through to be handled by first transformer
        "X_inner_mtype": CORE_MTYPES,
        "univariate-only": False,
    }

    # no further default tag values - these are set dynamically below

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_steps"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "steps_"

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_estimators(self.steps, cls_type=BaseTransformer)

        super().__init__()

        # abbreviate for readability
        ests = self.steps_
        first_trafo = ests[0][1]

        # input mtype and input type are as of the first estimator
        self.clone_tags(first_trafo, ["scitype:transform-input"])
        # chain requires X if and only if first estimator requires X
        self.clone_tags(first_trafo, ["requires_X"])
        # output type is that of last estimator, if no "Primitives" occur in the middle
        # if "Primitives" occur in the middle, then output is set to that too
        # this is in a case where "Series-to-Series" is applied to primitive df
        #   e.g., in a case of pipelining with scikit-learn transformers
        last_out = self._trafo_out()
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
        self._anytagis_then_set("requires_y", True, False, ests)

        # self can inverse transform if for all est, we either skip or can inv-transform
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
            not nested, contains only non-TransformerPipeline ``sktime`` transformers
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

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        TransformerPipeline object, concatenation of ``other`` (first) with ``self``
        (last).
            not nested, contains only non-TransformerPipeline ``sktime`` steps
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
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
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

    def _to_dim(self, x):
        """Translate scitype:transform-input or output tag to data dimension.

        Parameters
        ----------
        x : str, one of "Series", "Panel", "Hierarchical"
            scitype:transform-input or output tag

        Returns
        -------
        int
            data dimension corresponding to x
        """
        if x == "Series":
            return 1
        elif x == "Panel":
            return 2
        else:
            return 3

    def _dim_diff(self, obj):
        """Compute difference between input and output dimension."""
        inp = obj.get_tag("scitype:transform-input")
        out = obj.get_tag("scitype:transform-output")
        return self._to_dim(out) - self._to_dim(inp)

    def _dim_to_sci(self, d):
        """Translate data dimension to scitype:transform-output tag.

        Parameters
        ----------
        d : int
            data dimension

        Returns
        -------
        str
            scitype:transform-output tag corresponding to data dimension
        """
        if d <= 1:
            return "Series"
        elif d == 2:
            return "Panel"
        else:
            return "Hierarchical"

    def _trafo_out(self):
        """Infer scitype:transform-output tag.

        Uses the self.steps_ attribute, assumes it is initialized already.
        """
        ests = self.steps_
        est_list = [x[1] for x in ests]
        inp_dim = self._to_dim(est_list[0].get_tag("scitype:transform-input"))
        out_dim = inp_dim
        for est in est_list:
            dim_diff = self._dim_diff(est)
            out_dim = out_dim + dim_diff
        return self._dim_to_sci(out_dim)
