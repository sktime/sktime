# -*- coding: utf-8 -*-
"""Meta-transformers for building composite transformers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from enum import unique
from sklearn.base import clone

from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations.base import BaseTransformer

__author__ = ["fkiraly"]
__all__ = ["TransformerPipeline"]


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
    transformer : list of sktime transformers, or
        list of tuples (str, transformer) of sktime transformers
    """

    _required_parameters = ["transformers"]

    # no default tag values - these are set dynamically below

    def __init__(self, transformers):

        self.transformers = transformers
        self.transformers_ = self._check_estimators(transformers)

        super(TransformerPipeline, self).__init__()

        first_trafo = self.transformers_[0][1]
        last_trafo = self.transformers_[-1][1]

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
    def _transformers(self):
        return self._get_estimator_tuples(self.transformers, clone_ests=False)

    @_transformers.setter
    def _transformers(self, value):
        self.transformers = value

    def __mul__(self, other):
        """Magic * method, return (right) concatenated TransformerPipeline."""
        # we don't use names but _get_names to get the *original* names
        #   to avoid multiple "make unique" calls which may grow strings too much
        _, trafos = zip(*self.transformers_)
        names = tuple(self._get_orig_names(self.transformers))
        if isinstance(other, TransformerPipeline):
            _, trafos_o = zip(*other.transformers_)
            names_o = other._get_orig_names(other.transformers)
            new_names = names + names_o
            new_trafos = trafos + trafos_o
        elif isinstance(other, BaseTransformer):
            new_names = names + (type(other).__name__,)
            new_trafos = trafos + (other,)
        elif self._is_name_and_trafo(other):
            other_name = other[0]
            other_trafo = other[1]
            new_names = names + (other_name,)
            new_trafos = trafos + (other_trafo,)
        else:
            return NotImplemented

        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_trafos)):
            return TransformerPipeline(transformers=list(new_trafos))
        else:
            return TransformerPipeline(transformers=list(zip(new_names, new_trafos)))

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated TransformerPipeline."""
        _, trafos = zip(*self.transformers_)
        names = tuple(self._get_orig_names(self.transformers))
        if isinstance(other, TransformerPipeline):
            _, trafos_o = zip(*other.transformers_)
            names_o = other._get_orig_names(other.transformers)
            new_names = names_o + names
            new_trafos = trafos_o + trafos
        elif isinstance(other, BaseTransformer):
            new_names = (type(other).__name__,) + names
            new_trafos = (other,) + trafos
        elif self._is_name_and_trafo(other):
            other_name = other[0]
            other_trafo = other[1]
            new_names = (other_name,) + names
            new_trafos = (other_trafo,) + trafos
        else:
            return NotImplemented

        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_trafos)):
            return TransformerPipeline(transformers=list(new_trafos))
        else:
            return TransformerPipeline(transformers=list(zip(new_names, new_trafos)))

    @staticmethod
    def _is_name_and_trafo(obj):
        if not isinstance(obj, tuple) or len(obj) != 2:
            return False
        if not isinstance(obj[0], str) or not isinstance(obj[1], BaseTransformer):
            return False
        return True

    def _make_strings_unique(self, strlist):
        """Make a list or tuple of strings unique by appending _int of occurrence."""
        # if already unique, just return
        if len(set(strlist)) == len(strlist):
            return strlist

        # we convert internally to list, but remember whether it was tuple
        if isinstance(strlist, tuple):
            strlist = list(strlist)
            was_tuple = True
        else:
            was_tuple = False

        from collections import Counter

        strcount = Counter(strlist)

        # if any duplicates, we append _integer of occurrence to non-uniques
        nowcount = Counter()
        uniquestr = strlist
        for i, x in enumerate(uniquestr):
            if strcount[x] > 1:
                nowcount.update([x])
                uniquestr[i] = x + "_" + str(nowcount[x])

        if was_tuple:
            uniquestr = tuple(uniquestr)

        # repeat until all are unique
        #   the algorithm recurses, but will always terminate
        #   because potential clashes are lexicographically increasing
        return self._make_strings_unique(uniquestr)

    def _anytagis(self, tag_name, value):
        """Return whether any estimator in list has tag `tag_name` of value `value`."""
        tagis = [est.get_tag(tag_name, value) == value for _, est in self.transformers_]
        return any(tagis)

    def _anytagis_then_set(self, tag_name, value, value_if_not):
        """Set self's `tag_name` tag to `value` if any estimator on the list has it."""
        if self._anytagis(tag_name=tag_name, value=value):
            self.set_tags(**{tag_name: value})
        else:
            self.set_tags(**{tag_name: value_if_not})

    def _anytag_notnone_val(self, tag_name):
        """Return first non-'None' value of tag `tag_name` in estimator list."""
        for _, est in self.transformers_:
            tag_val = est.get_tag(tag_name)
            if tag_val != "None":
                return tag_val
        return tag_val

    def _anytag_notnone_set(self, tag_name):
        """Set self's `tag_name` tag to first non-'None' value in estimator list."""
        tag_val = self._anytag_notnone_val(tag_name=tag_name)
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
        Xt = X
        for _, transformer in self.transformers_:
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
        """Get parameters of estimator in `transformers`.

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
        return self._get_params("_transformers", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `transformers`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params("_transformers", **kwargs)
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

        return self._get_estimator_tuples(estimators, clone_ests=True)

    def _get_estimator_list(self, estimators):
        """Return list of estimators, from a list or tuple.

        Arguments
        ---------
        estimators : list of estimators, or list of (str, estimator tuples)

        Returns
        -------
        list of estimators - identical with estimators if list of estimators
            if list of (str, estimator) tuples, the str get removed
        """
        if isinstance(estimators[0], tuple):
            return [x[1] for x in estimators]
        else:
            return estimators

    def _make_names(self, estimators):
        """Return *unique*-made names for the estimators.

        Arguments
        ---------
        estimators : list of estimators, or list of (str, estimator tuples)

        Returns
        -------
        unique_names : list of str, unique entries, of equal length as estimators
            list of unique names for estimators, made unique using _make_strings_unique 
        """
        if isinstance(estimators[0], tuple):
            names = [x[0] for x in estimators]
        else:
            names = [type(e).__name__ for e in estimators]
        unique_names = self._make_strings_unique(names)
        return unique_names

    def _get_estimator_tuples(self, estimators, clone_ests=False):
        """Return list of estimator tuples, from a list or tuple.

        Arguments
        ---------
        estimators : list of estimators, or list of (str, estimator tuples)
        clone_ests : bool, whether estimators get cloned in the process

        Returns
        -------
        est_tuples : list of (str, estimator) tuples
            if estimators was a list of (str, estimator) tuples, then identical/cloned
            if was a list of estimators, then str are generated via _name_names
        """
        ests = self._get_estimator_list(estimators)
        if clone_ests:
            ests = [clone(e) for e in ests]
        unique_names = self._make_names(estimators)
        est_tuples = list(zip(unique_names, ests))
        return est_tuples

    def _get_orig_names(self, estimators):
        """Return original, potentially non-unique names for the estimators.

        Arguments
        ---------
        estimators : list of estimators, or list of (str, estimator tuples)

        Returns
        -------
        names : list of str, unique entries, of equal length as estimators
        """
        if estimators is None or len(estimators) == 0:
            names = []
        if isinstance(estimators[0], BaseTransformer):
            names = [type(e).__name__ for e in estimators]
        if isinstance(estimators[0], tuple):
            names = [e[0] for e in estimators]
        return tuple(names)

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
