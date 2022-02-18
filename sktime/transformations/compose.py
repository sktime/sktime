# -*- coding: utf-8 -*-
"""Meta-transformers for building composite transformers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations.base import BaseTransformer

__author__ = ["fkiraly"]
__all__ = ["TransformerPipeline"]


class TransformerPipeline(BaseTransformer, _HeterogenousMetaEstimator):
    """Wrap an existing transformer to tune whether to include it in a pipeline.

    Allows tuning the implicit hyperparameter whether or not to use a
    particular transformer inside a pipeline (e.g. TranformedTargetForecaster)
    or not. This is achieved by the hyperparameter `passthrough`
    which can be added to a tuning grid then (see example).

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series.
    passthrough : bool, default=False
       Whether to apply the given transformer or to just
        passthrough the data (identity transformation). If, True the transformer
        is not applied and the OptionalPassthrough uses the identity
        transformation.
    """

    _required_parameters = ["transformers"]

    # no default tag values - these are set dynamically below

    def __init__(self, transformers):

        self.transformers = transformers
        self.estimators = self._check_estimators(transformers)

        super(TransformerPipeline, self).__init__()

        first_trafo = self.estimators[0][1]
        last_trafo = self.estimators[-1][1]

        self.clone_tags(first_trafo, ["X_inner_mtype", "scitype:transform-input"])
        self.clone_tags(last_trafo, "scitype:transform-output")

        self._anytag_notnone_set("y_inner_mtype")
        self._anytag_notnone_set("scitype:transform-labels")

        self._anytagis_then_set("scitype:instancewise", False, True)
        self._anytagis_then_set("X-y-must-have-same-index", True, False)
        self._anytagis_then_set("fit-in-transform", False, True)
        self._anytagis_then_set("transform-returns-same-time-index", False, True)
        self._anytagis_then_set("skip-inverse-transform", True, False)
        self._anytagis_then_set("capability:inverse_transform", False, True)
        self._anytagis_then_set("handles-missing-data", False, True)
        self._anytagis_then_set("univariate-only", True, False)

    @property
    def estimators(self):
        """Get estimators named list."""
        return self._estimators

    @estimators.setter
    def estimators(self, value):
        """Set estimators named list."""
        estimators = value
        if isinstance(estimators[0], tuple):
            est_list = estimators
        else:
            names = [type(x).__name__ for x in estimators]
            unique_names = self._make_strings_unique(names)
            est_list = [(unique_names[i], t) for i, t in enumerate(estimators)]
        self._estimators = est_list


    @staticmethod
    def _make_strings_unique(strlist):

        from collections import Counter
        strcount = Counter(strlist)

        nowcount = Counter()
        uniquestr = strlist
        for i, x in enumerate(uniquestr):
            if strcount[x] > 1:
                nowcount.update(x)
                uniquestr[i] = x + "_" + str(nowcount[x])

        return uniquestr

    def _anytagis(self, tag_name, value):
        """Return whether any estimator in list has tag `tag_name` of value `value`."""
        tagis = [est.get_tag(tag_name, value) for _, est in self._estimators]
        return any(tagis)

    def _anytagis_then_set(self, tag_name, value, value_if_not):
        """Set self's `tag_name` tag to `value` if any estimator on the list has it."""
        if self._anytagis(tag_name=tag_name, value=value):
            self.set_tags(**{tag_name: value})
        else:
            self.set_tags(**{tag_name: value_if_not})

    def _anytag_notnone_val(self, tag_name):
        """Return first non-'None' value of tag `tag_name` in estimator list."""
        for _, est in self._estimators:
            tag_val = est.get_tag(tag_name)
            if tag_val != "None":
                return tag_val
        return tag_val

    def _anytag_notnone_set(self, tag_name):
        """Set self's `tag_name` tag to first non-'None' value in estimator list."""
        tag_val = self._anytag_notnone_val
        if tag_val != "None":
            self.set_tags(**{tag_name: tag_val})

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
        self.transformers_ = []
        Xt = X

        for (name, transformer) in self.estimators:
            transformer_ = clone(transformer)
            Xt = transformer_.fit_transform(X=Xt, y=y)
            self.transformers_.append((name, transformer_))

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
        for _, transformer in self.transformers_:
            Xt = transformer.transform(X=Xt, y=y)

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
        for _, transformer in self.transformers_:
            Xt = transformer.inverse_transform(X=Xt, y=y)

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
        for _, transformer in self.transformers_:
            transformer.update(X=Xt, y=y)
            Xt = transformer.transform(X=Xt, y=y)

        return self

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
        return self._get_params("transformers_", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `_forecasters`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params("transformers_", **kwargs)
        return self

    def _check_estimators(self, estimators, attr_name="transformers"):

        msg = (
            f"Invalid '{attr_name}' attribute, '{attr_name}' should be a list"
            " of estimators, or a list of (string, estimator) tuples."
        )

        if (
            estimators is None
            or len(estimators) == 0
            or not isinstance(estimators, list)
        ):
            raise TypeError(msg)

        if not isinstance(estimators[0], (BaseTransformer, tuple)):
            raise TypeError(msg)

        if isinstance(estimators[0], BaseTransformer):
            if not all(isinstance(est, BaseTransformer) for est in estimators):
                raise TypeError(msg)
        if isinstance(estimators[0], tuple):
            if not all(isinstance(est, tuple) for est in estimators):
                raise TypeError(msg)
            if not all(isinstance(est[0], str) for est in estimators):
                raise TypeError(msg)
            if not all(isinstance(est[1], BaseTransformer) for est in estimators):
                raise TypeError(msg)
            names, _ = zip(*estimators)
            # defined by MetaEstimatorMixin
            self._check_names(names)

        return estimators

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

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

        trafo1 = ExponentTransformer(power=2)
        trafo2 = ExponentTransformer(power=0.5)
        params = {"transformers": [trafo1, trafo2]}
        return params
