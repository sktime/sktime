# -*- coding: utf-8 -*-
"""Meta-transformers for building composite transformers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

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
        self.steps_ = self._check_estimators(steps)

        super(TransformerPipeline, self).__init__()

        first_trafo = self.steps_[0][1]
        last_trafo = self.steps_[-1][1]

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
        # we don't use names but _get_estimator_names to get the *original* names
        #   to avoid multiple "make unique" calls which may grow strings too much
        _, trafos = zip(*self.steps_)
        names = tuple(self._get_estimator_names(self.steps))
        if isinstance(other, TransformerPipeline):
            _, trafos_o = zip(*other.steps_)
            names_o = tuple(other._get_estimator_names(other.steps))
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
            return TransformerPipeline(steps=list(new_trafos))
        else:
            return TransformerPipeline(steps=list(zip(new_names, new_trafos)))

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
        _, trafos = zip(*self.steps_)
        names = tuple(self._get_estimator_names(self.steps))
        if isinstance(other, TransformerPipeline):
            _, trafos_o = zip(*other.steps_)
            names_o = tuple(other._get_estimator_names(other.steps))
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
            return TransformerPipeline(steps=list(new_trafos))
        else:
            return TransformerPipeline(steps=list(zip(new_names, new_trafos)))

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
        tagis = [est.get_tag(tag_name, value) == value for _, est in self.steps_]
        return any(tagis)

    def _anytagis_then_set(self, tag_name, value, value_if_not):
        """Set self's `tag_name` tag to `value` if any estimator on the list has it."""
        if self._anytagis(tag_name=tag_name, value=value):
            self.set_tags(**{tag_name: value})
        else:
            self.set_tags(**{tag_name: value_if_not})

    def _anytag_notnone_val(self, tag_name):
        """Return first non-'None' value of tag `tag_name` in estimator list."""
        for _, est in self.steps_:
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
            if not self.get_tag("fit-in-transform", False):
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
        for _, transformer in self.steps_:
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
        for _, transformer in self.steps_:
            transformer.update(X=Xt, y=y)
            Xt = transformer.transform(X=Xt, y=y)

        return self

    def get_params(self, deep=True):
        """Get parameters of estimator in `steps`.

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

    def _check_estimators(self, estimators, attr_name="steps"):

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

    def _get_estimator_names(self, estimators, make_unique=False):
        """Return names for the estimators, optionally made unique.

        Arguments
        ---------
        estimators : list of estimators, or list of (str, estimator tuples)
        make_unique : bool, optional, default=False
            whether names should be made unique in the return

        Returns
        -------
        names : list of str, unique entries, of equal length as estimators
            names for estimators in estimators
            if make_unique=True, made unique using _make_strings_unique
        """
        if estimators is None or len(estimators) == 0:
            names = []
        elif isinstance(estimators[0], tuple):
            names = [x[0] for x in estimators]
        elif isinstance(estimators[0], BaseTransformer):
            names = [type(e).__name__ for e in estimators]
        else:
            raise RuntimeError(
                "unreachable condition in _get_estimator_names, "
                " likely input assumptions are violated,"
                " run _check_estimators before running _get_estimator_names"
            )
        if make_unique:
            names = self._make_strings_unique(names)
        return names

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
        unique_names = self._get_estimator_names(estimators, make_unique=True)
        est_tuples = list(zip(unique_names, ests))
        return est_tuples

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
