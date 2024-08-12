#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements meta estimator for estimators composed of other estimators."""

__author__ = ["mloning", "fkiraly"]
__all__ = ["_HeterogenousMetaEstimator", "_ColumnEstimator"]

from inspect import isclass

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator


class _HeterogenousMetaEstimator:
    """Handles parameter management for estimators composed of named estimators.

    Partly adapted from sklearn utils.metaestimator.py.
    """

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

    def get_params(self, deep=True):
        """Get parameters of estimator.

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
        steps = self._steps_attr
        return self._get_params(steps, deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        steps_attr = self._steps_attr
        self._set_params(steps_attr, **kwargs)
        return self

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        fitted_params = self._get_fitted_params_default()

        steps = self._steps_fitted_attr
        steps_params = self._get_params(steps, fitted=True)

        fitted_params.update(steps_params)

        return fitted_params

    def is_composite(self):
        """Check if the object is composite.

        A composite object is an object which contains objects, as parameters.
        Called on an instance, since this may differ by instance.

        Returns
        -------
        composite: bool, whether self contains a parameter which is BaseObject
        """
        # children of this class are always composite
        return True

    def _get_params(self, attr, deep=True, fitted=False):
        if fitted:
            private_method = "_get_fitted_params"
            public_method = "get_fitted_params"
            deepkw = {}
        else:
            private_method = "get_params"
            public_method = "get_params"
            deepkw = {"deep": deep}

        out = getattr(super(), private_method)(**deepkw)
        if deep and hasattr(self, attr):
            estimators = getattr(self, attr)
            estimators = [(x[0], x[1]) for x in estimators]
            out.update(estimators)
            for name, estimator in estimators:
                # checks estimator has the method we want to call
                cond1 = hasattr(estimator, public_method)
                # checks estimator is fitted if calling get_fitted_params
                is_fitted = hasattr(estimator, "is_fitted") and estimator.is_fitted
                # if we call get_params and not get_fitted_params, this is True
                cond2 = not fitted or is_fitted
                # check both conditions together
                if cond1 and cond2:
                    for key, value in getattr(estimator, public_method)(
                        **deepkw
                    ).items():
                        out[f"{name}__{key}"] = value
        return out

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        items = getattr(self, attr)
        names = []
        if items and isinstance(items, (list, tuple)):
            names = list(zip(*items))[0]
        for name in list(params.keys()):
            if "__" not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, est_tpl in enumerate(new_estimators):
            estimator_name = est_tpl[0]
            if estimator_name == name:
                new_tpl = list(est_tpl)
                new_tpl[1] = new_val
                new_estimators[i] = tuple(new_tpl)
                break
        setattr(self, attr, new_estimators)

    def _check_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError(f"Names provided are not unique: {list(names)!r}")
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                "Estimator names conflict with constructor "
                f"arguments: {sorted(invalid_names)!r}"
            )
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                "Estimator names must not contain __: got " f"{invalid_names!r}"
            )

    def _subset_dict_keys(self, dict_to_subset, keys, prefix=None):
        """Subset dictionary d to keys in keys.

        Subsets `dict_to_subset` to keys in iterable `keys`.

        If `prefix` is passed, subsets to `f"{prefix}__{key}"` for all `key` in `keys`.
        The prefix is then removed from the keys of the return dict, i.e.,
        return has keys `{key}` where `f"{prefix}__{key}"` was key in `dict_to_subset`.
        Note that passing `prefix` will turn non-str keys into str keys.

        Parameters
        ----------
        dict_to_subset : dict
            dictionary to subset by keys
        keys : iterable
        prefix : str or None, optional

        Returns
        -------
        `subsetted_dict` : dict
            `dict_to_subset` subset to keys in `keys` described as above
        """

        def rem_prefix(x):
            if prefix is None:
                return x
            prefix__ = f"{prefix}__"
            if x.startswith(prefix__):
                return x[len(prefix__) :]
            else:
                return x

        if prefix is not None:
            keys = [f"{prefix}__{key}" for key in keys]
        keys_in_both = set(keys).intersection(dict_to_subset.keys())
        subsetted_dict = {rem_prefix(k): dict_to_subset[k] for k in keys_in_both}
        return subsetted_dict

    @staticmethod
    def _is_name_and_est(obj, cls_type=None):
        """Check whether obj is a tuple of type (str, cls_type).

        Parameters
        ----------
        cls_type : class or tuple of class, optional. Default = BaseEstimator.
            class(es) that all estimators are checked to be an instance of

        Returns
        -------
        bool : True if obj is (str, cls_type) tuple, False otherwise
        """
        if cls_type is None:
            cls_type = BaseEstimator
        if not isinstance(obj, tuple) or len(obj) != 2:
            return False
        if not isinstance(obj[0], str) or not isinstance(obj[1], cls_type):
            return False
        return True

    def _check_estimators(
        self,
        estimators,
        attr_name="steps",
        cls_type=None,
        allow_mix=True,
        clone_ests=True,
    ):
        """Check that estimators is a list of estimators or list of str/est tuples.

        Parameters
        ----------
        estimators : any object
            should be list of estimators or list of (str, estimator) tuples
            estimators should inherit from cls_type class
        attr_name : str, optional. Default = "steps"
            Name of checked attribute in error messages
        cls_type : class or tuple of class, optional. Default = BaseEstimator.
            class(es) that all estimators are checked to be an instance of
        allow_mix : boolean, optional. Default = True.
            whether mix of estimator and (str, estimator) is allowed in `estimators`
        clone_ests : boolean, optional. Default = True.
            whether estimators in return are cloned (True) or references (False).

        Returns
        -------
        est_tuples : list of (str, estimator) tuples
            if estimators was a list of (str, estimator) tuples, then identical/cloned
            if was a list of estimators, then str are generated via _get_estimator_names

        Raises
        ------
        TypeError, if estimators is not a list of estimators or (str, estimator) tuples
        TypeError, if estimators in the list are not instances of cls_type
        """
        msg = (
            f"Invalid {attr_name!r} attribute, {attr_name!r} should be a list"
            " of estimators, or a list of (string, estimator) tuples. "
        )
        if cls_type is None:
            msg += f"All estimators in {attr_name!r} must be of type BaseEstimator."
            cls_type = BaseEstimator
        elif isclass(cls_type) or isinstance(cls_type, tuple):
            msg += (
                f"All estimators in {attr_name!r} must be of type "
                f"{cls_type.__name__}."
            )
        else:
            raise TypeError("cls_type must be a class or tuple of classes")

        if (
            estimators is None
            or len(estimators) == 0
            or not isinstance(estimators, list)
        ):
            raise TypeError(msg)

        def is_est_is_tuple(obj):
            """Check whether obj is estimator of right type, or (str, est) tuple."""
            is_est = isinstance(obj, cls_type)
            is_tuple = self._is_name_and_est(obj, cls_type)

            return is_est, is_tuple

        if not all(any(is_est_is_tuple(x)) for x in estimators):
            raise TypeError(msg)

        msg_no_mix = (
            f"elements of {attr_name} must either all be estimators, "
            f"or all (str, estimator) tuples, mix of the two is not allowed"
        )

        if not allow_mix and not all(is_est_is_tuple(x)[0] for x in estimators):
            if not all(is_est_is_tuple(x)[1] for x in estimators):
                raise TypeError(msg_no_mix)

        return self._get_estimator_tuples(estimators, clone_ests=clone_ests)

    def _coerce_estimator_tuple(self, obj, clone_est=False):
        """Coerce estimator or (str, estimator) tuple to (str, estimator) tuple.

        Parameters
        ----------
        obj : estimator or (str, estimator) tuple
            assumes that this has been checked, no checks are performed
        clone_est : boolean, optional. Default = False.
            Whether to return clone of estimator in obj (True) or a reference (False).

        Returns
        -------
        est_tuple : (str, stimator tuple)
            obj if obj was (str, estimator) tuple
            (obj class name, obj) if obj was estimator
        """
        if isinstance(obj, tuple):
            est = obj[1]
            name = obj[0]
        else:
            est = obj
            name = type(obj).__name__

        if clone_est:
            return (name, est.clone())
        else:
            return (name, est)

    def _get_estimator_list(self, estimators):
        """Return list of estimators, from a list or tuple.

        Parameters
        ----------
        estimators : list of estimators, or list of (str, estimator tuples)

        Returns
        -------
        list of estimators - identical with estimators if list of estimators
            if list of (str, estimator) tuples, the str get removed
        """
        return [self._coerce_estimator_tuple(x)[1] for x in estimators]

    def _get_estimator_names(self, estimators, make_unique=False):
        """Return names for the estimators, optionally made unique.

        Parameters
        ----------
        estimators : list of estimators, or list of (str, estimator tuples)
        make_unique : bool, optional, default=False
            whether names should be made unique in the return

        Returns
        -------
        names : list of str, unique entries, of equal length as estimators
            names for estimators in estimators
            if make_unique=True, made unique using _make_strings_unique
        """
        names = [self._coerce_estimator_tuple(x)[0] for x in estimators]
        if make_unique:
            names = self._make_strings_unique(names)
        return names

    def _get_estimator_tuples(self, estimators, clone_ests=False):
        """Return list of estimator tuples, from a list or tuple.

        Parameters
        ----------
        estimators : list of estimators, or list of (str, estimator tuples)
        clone_ests : bool, optional, default=False.
            whether estimators of the return are cloned (True) or references (False)

        Returns
        -------
        est_tuples : list of (str, estimator) tuples
            if estimators was a list of (str, estimator) tuples, then identical/cloned
            if was a list of estimators, then str are generated via _get_estimator_names
        """
        ests = self._get_estimator_list(estimators)
        if clone_ests:
            ests = [e.clone() for e in ests]
        unique_names = self._get_estimator_names(estimators, make_unique=True)
        est_tuples = list(zip(unique_names, ests))
        return est_tuples

    def _make_strings_unique(self, strlist):
        """Make a list or tuple of strings unique by appending _int of occurrence.

        Parameters
        ----------
        strlist : nested list/tuple structure with string elements

        Returns
        -------
        uniquestr : nested list/tuple structure with string elements
            has same bracketing as `strlist`
            string elements, if not unique, are replaced by unique strings
                if any duplicates, _integer of occurrence is appended to non-uniques
                e.g., "abc", "abc", "bcd" becomes "abc_1", "abc_2", "bcd"
                in case of clashes, process is repeated until it terminates
                e.g., "abc", "abc", "abc_1" becomes "abc_0", "abc_1_0", "abc_1_1"
        """
        # recursions to guarantee that strlist is flat list of strings
        ##############################################################

        # if strlist is not flat, flatten and apply, then unflatten
        if not is_flat(strlist):
            flat_strlist = flatten(strlist)
            unique_flat_strlist = self._make_strings_unique(flat_strlist)
            uniquestr = unflatten(unique_flat_strlist, strlist)
            return uniquestr

        # now we can assume that strlist is flat

        # if strlist is a tuple, convert to list, apply this function, then convert back
        if isinstance(strlist, tuple):
            uniquestr = self._make_strings_unique(list(strlist))
            uniquestr = tuple(strlist)
            return uniquestr

        # end of recursions
        ###################
        # now we can assume that strlist is a flat list

        # if already unique, just return
        if len(set(strlist)) == len(strlist):
            return strlist

        from collections import Counter

        strcount = Counter(strlist)

        # if any duplicates, we append _integer of occurrence to non-uniques
        nowcount = Counter()
        uniquestr = strlist
        for i, x in enumerate(uniquestr):
            if strcount[x] > 1:
                nowcount.update([x])
                uniquestr[i] = x + "_" + str(nowcount[x])

        # repeat until all are unique
        #   the algorithm recurses, but will always terminate
        #   because potential clashes are lexicographically increasing
        return self._make_strings_unique(uniquestr)

    def _dunder_concat(
        self,
        other,
        base_class,
        composite_class,
        attr_name="steps",
        concat_order="left",
        composite_params=None,
    ):
        """Concatenate pipelines for dunder parsing, helper function.

        This is used in concrete heterogeneous meta-estimators that implement
        dunders for easy concatenation of pipeline-like composites.
        Examples: TransformerPipeline, MultiplexForecaster, FeatureUnion

        Parameters
        ----------
        self : `sktime` estimator, instance of composite_class (when this is invoked)
        other : `sktime` estimator, should inherit from composite_class or base_class
            otherwise, `NotImplemented` is returned
        base_class : estimator base class assumed as base class for self, other,
            and estimator components of composite_class, in case of concatenation
        composite_class : estimator class that has attr_name attribute in instances
            attr_name attribute should contain list of base_class estimators,
            list of (str, base_class) tuples, or a mixture thereof
        attr_name : str, optional, default="steps"
            name of the attribute that contains estimator or (str, estimator) list
            concatenation is done for this attribute, see below
        concat_order : str, one of "left" and "right", optional, default="left"
            if "left", result attr_name will be like self.attr_name + other.attr_name
            if "right", result attr_name will be like other.attr_name + self.attr_name
        composite_params : dict, optional, default=None; else, pairs strname-value
            if not None, parameters of the composite are always set accordingly
            i.e., contains key-value pairs, and composite_class has key set to value

        Returns
        -------
        instance of composite_class, where attr_name is a concatenation of
        self.attr_name and other.attr_name, if other was of composite_class
        if other is of base_class, then composite_class(attr_name=other) is used
        in place of other, for the concatenation
        concat_order determines which list is first, see above
        "concatenation" means: resulting instance's attr_name contains
        list of (str, est), a direct result of concat self.attr_name and other.attr_name
        if str are all the class names of est, list of est only is used instead
        """
        # input checks
        if not isinstance(concat_order, str):
            raise TypeError(f"concat_order must be str, but found {type(concat_order)}")
        if concat_order not in ["left", "right"]:
            raise ValueError(
                f'concat_order must be one of "left", "right", but found '
                f"{concat_order!r}"
            )
        if not isinstance(attr_name, str):
            raise TypeError(f"attr_name must be str, but found {type(attr_name)}")
        if not isclass(composite_class):
            raise TypeError("composite_class must be a class")
        if not isclass(base_class):
            raise TypeError("base_class must be a class")
        if not issubclass(composite_class, base_class):
            raise ValueError("composite_class must be a subclass of base_class")
        if not isinstance(self, composite_class):
            raise TypeError("self must be an instance of composite_class")

        def concat(x, y):
            if concat_order == "left":
                return x + y
            else:
                return y + x

        # get attr_name from self and other
        # can be list of ests, list of (str, est) tuples, or list of miture
        self_attr = getattr(self, attr_name)

        # from that, obtain ests, and original names (may be non-unique)
        # we avoid _make_strings_unique call too early to avoid blow-up of string
        ests_s = tuple(self._get_estimator_list(self_attr))
        names_s = tuple(self._get_estimator_names(self_attr))
        if isinstance(other, composite_class):
            other_attr = getattr(other, attr_name)
            ests_o = tuple(other._get_estimator_list(other_attr))
            names_o = tuple(other._get_estimator_names(other_attr))
            new_names = concat(names_s, names_o)
            new_ests = concat(ests_s, ests_o)
        elif isinstance(other, base_class):
            new_names = concat(names_s, (type(other).__name__,))
            new_ests = concat(ests_s, (other,))
        elif self._is_name_and_est(other, base_class):
            other_name = other[0]
            other_est = other[1]
            new_names = concat(names_s, (other_name,))
            new_ests = concat(ests_s, (other_est,))
        else:
            return NotImplemented

        # create the "steps" param for the composite
        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_ests)):
            step_param = {attr_name: list(new_ests)}
        else:
            step_param = {attr_name: list(zip(new_names, new_ests))}

        # retrieve other parameters, from composite_params attribute
        if composite_params is None:
            composite_params = {}
        else:
            composite_params = composite_params.copy()

        # construct the composite with both step and additional params
        composite_params.update(step_param)
        return composite_class(**composite_params)

    def _anytagis(self, tag_name, value, estimators):
        """Return whether any estimator in list has tag `tag_name` of value `value`.

        Parameters
        ----------
        tag_name : str, name of the tag to check
        value : value of the tag to check for
        estimators : list of (str, estimator) pairs to query for the tag/value

        Return
        ------
        bool : True iff at least one estimator in the list has value in tag tag_name
        """
        tagis = [est.get_tag(tag_name, value) == value for _, est in estimators]
        return any(tagis)

    def _anytagis_then_set(self, tag_name, value, value_if_not, estimators):
        """Set self's `tag_name` tag to `value` if any estimator on the list has it.

        Writes to self:
        sets the tag `tag_name` to `value` if `_anytagis(tag_name, value)` is True
            otherwise sets the tag `tag_name` to `value_if_not`

        Parameters
        ----------
        tag_name : str, name of the tag
        value : value to check and to set tag to if one of the tag values is `value`
        value_if_not : value to set in self if none of the tag values is `value`
        estimators : list of (str, estimator) pairs to query for the tag/value
        """
        if self._anytagis(tag_name=tag_name, value=value, estimators=estimators):
            self.set_tags(**{tag_name: value})
        else:
            self.set_tags(**{tag_name: value_if_not})

    def _anytag_notnone_val(self, tag_name, estimators):
        """Return first non-'None' value of tag `tag_name` in estimator list.

        Parameters
        ----------
        tag_name : str, name of the tag
        estimators : list of (str, estimator) pairs to query for the tag/value

        Return
        ------
        tag_val : first non-'None' value of tag `tag_name` in estimator list.
        """
        for _, est in estimators:
            tag_val = est.get_tag(tag_name)
            if tag_val != "None":
                return tag_val
        return tag_val

    def _anytag_notnone_set(self, tag_name, estimators):
        """Set self's `tag_name` tag to first non-'None' value in estimator list.

        Writes to self:
        tag with name tag_name, sets to _anytag_notnone_val(tag_name, estimators)

        Parameters
        ----------
        tag_name : str, name of the tag
        estimators : list of (str, estimator) pairs to query for the tag/value
        """
        tag_val = self._anytag_notnone_val(tag_name=tag_name, estimators=estimators)
        if tag_val != "None":
            self.set_tags(**{tag_name: tag_val})

    def _tagchain_is_linked(
        self,
        left_tag_name,
        mid_tag_name,
        estimators,
        left_tag_val=True,
        mid_tag_val=True,
    ):
        """Check whether all tags left of the first mid_tag/val are left_tag/val.

        Useful to check, for instance, whether all instances of estimators
            left of the first missing value imputer can deal with missing values.

        Parameters
        ----------
        left_tag_name : str, name of the left tag
        mid_tag_name : str, name of the middle tag
        estimators : list of (str, estimator) pairs to query for the tag/value
        left_tag_val : value of the left tag, optional, default=True
        mid_tag_val : value of the middle tag, optional, default=True

        Returns
        -------
        chain_is_linked : bool,
            True iff all "left" tag instances `left_tag_name` have value `left_tag_val`
            a "left" tag instance is an instance in estimators which is earlier
            than the first occurrence of `mid_tag_name` with value `mid_tag_val`
        chain_is_complete : bool,
            True iff chain_is_linked is True, and
                there is an occurrence of `mid_tag_name` with value `mid_tag_val`
        """
        for _, est in estimators:
            if est.get_tag(mid_tag_name) == mid_tag_val:
                return True, True
            if not est.get_tag(left_tag_name) == left_tag_val:
                return False, False
        return True, False

    def _tagchain_is_linked_set(
        self,
        left_tag_name,
        mid_tag_name,
        estimators,
        left_tag_val=True,
        mid_tag_val=True,
        left_tag_val_not=False,
        mid_tag_val_not=False,
    ):
        """Check if _tagchain_is_linked, then set self left_tag_name and mid_tag_name.

        Writes to self:
        tag with name left_tag_name, sets to left_tag_val if _tag_chain_is_linked[0]
            otherwise sets to left_tag_val_not
        tag with name mid_tag_name, sets to mid_tag_val if _tag_chain_is_linked[1]
            otherwise sets to mid_tag_val_not

        Parameters
        ----------
        left_tag_name : str, name of the left tag
        mid_tag_name : str, name of the middle tag
        estimators : list of (str, estimator) pairs to query for the tag/value
        left_tag_val : value of the left tag, optional, default=True
        mid_tag_val : value of the middle tag, optional, default=True
        left_tag_val_not : value to set if not linked, optional, default=False
        mid_tag_val_not : value to set if not linked, optional, default=False
        """
        linked, complete = self._tagchain_is_linked(
            left_tag_name=left_tag_name,
            mid_tag_name=mid_tag_name,
            estimators=estimators,
            left_tag_val=left_tag_val,
            mid_tag_val=mid_tag_val,
        )
        if linked:
            self.set_tags(**{left_tag_name: left_tag_val})
        else:
            self.set_tags(**{left_tag_name: left_tag_val_not})
        if complete:
            self.set_tags(**{mid_tag_name: mid_tag_val})
        else:
            self.set_tags(**{mid_tag_name: mid_tag_val_not})


def flatten(obj):
    """Flatten nested list/tuple structure.

    Parameters
    ----------
    obj: nested list/tuple structure

    Returns
    -------
    list or tuple, tuple if obj was tuple, list otherwise
        flat iterable, containing non-list/tuple elements in obj in same order as in obj

    Example
    -------
    >>> flatten([1, 2, [3, (4, 5)], 6])
    [1, 2, 3, 4, 5, 6]
    """
    if not isinstance(obj, (list, tuple)):
        return [obj]
    else:
        return type(obj)([y for x in obj for y in flatten(x)])


def unflatten(obj, template):
    """Invert flattening, given template for nested list/tuple structure.

    Parameters
    ----------
    obj : list or tuple of elements
    template : nested list/tuple structure
        number of non-list/tuple elements of obj and template must be equal

    Returns
    -------
    rest : list or tuple of elements
        has element bracketing exactly as `template`
            and elements in sequence exactly as `obj`

    Example
    -------
    >>> unflatten([1, 2, 3, 4, 5, 6], [6, 3, [5, (2, 4)], 1])
    [1, 2, [3, (4, 5)], 6]
    """
    if not isinstance(template, (list, tuple)):
        return obj[0]

    list_or_tuple = type(template)
    ls = [unflat_len(x) for x in template]
    for i in range(1, len(ls)):
        ls[i] += ls[i - 1]
    ls = [0] + ls

    res = [unflatten(obj[ls[i] : ls[i + 1]], template[i]) for i in range(len(ls) - 1)]

    return list_or_tuple(res)


def unflat_len(obj):
    """Return number of non-list/tuple elements in obj."""
    if not isinstance(obj, (list, tuple)):
        return 1
    else:
        return sum([unflat_len(x) for x in obj])


def is_flat(obj):
    """Check whether list or tuple is flat, returns true if yes, false if nested."""
    return not any(isinstance(x, (list, tuple)) for x in obj)


class _ColumnEstimator:
    """Mixin class with utilities for by-column applicates."""

    def _coerce_to_pd_index(self, obj, ref=None):
        """Coerce obj to pandas Index, replacing ints by index elements.

        Parameters
        ----------
        obj : iterable of pandas compatible index elements or int
        ref : reference index, coercible to pd.Index, optional, default=None

        Returns
        -------
        obj coerced to pd.Index
            if ref was passed, and
            if obj had int or np.integer elements which do not occur in ref,
            then each int-like element is replaced by the i-th element of ref
        """
        if ref is not None:
            # coerce ref to pd.Index
            if not isinstance(ref, pd.Index):
                if hasattr(ref, "columns") and isinstance(ref.index, pd.Index):
                    ref = ref.columns
            else:
                ref = pd.Index(ref)

            # replace ints by column names
            obj = self._get_indices(ref, obj)

        # deal with numpy int by coercing to python int
        if np.issubdtype(type(obj), np.integer):
            obj = int(obj)

        # coerce to pd.Index
        if isinstance(obj, (int, str)):
            return pd.Index([obj])
        else:
            return pd.Index(obj)

    def _get_indices(self, y, idx):
        """Convert integer indices if necessary."""
        if hasattr(y, "columns"):
            y = y.columns

        def _get_index(y, ix):
            # deal with numpy int by coercing to python int
            if np.issubdtype(type(ix), np.integer):
                ix = int(ix)

            if isinstance(ix, int) and ix not in y and ix < len(y):
                return y[ix]
            else:
                return ix

        if isinstance(idx, (list, tuple)):
            return [self._get_indices(y, ix) for ix in idx]
        else:
            return _get_index(y, idx)

    def _by_column(self, methodname, **kwargs):
        """Apply self.methodname to kwargs by column, then column-concatenate.

        Parameters
        ----------
        methodname : str, one of the methods of self
            assumed to take kwargs and return pd.DataFrame
        col_multiindex : bool, optional, default=False
            if True, will add an additional column multiindex at top, entries = index

        Returns
        -------
        y_pred : pd.DataFrame
            result of [f.methodname(**kwargs) for _, f, _ in self.forecsaters_]
            column-concatenated with keys being the variable names last seen in y
        """
        # get col_multiindex arg from kwargs
        col_multiindex = kwargs.pop("col_multiindex", False)

        y_preds = []
        keys = []
        for _, est, index in getattr(self, self._steps_fitted_attr):
            y_preds += [getattr(est, methodname)(**kwargs)]
            keys += [index]

        keys = self._get_indices(self._y, keys)

        if col_multiindex:
            y_pred = pd.concat(y_preds, axis=1, keys=keys)
        else:
            y_pred = pd.concat(y_preds, axis=1)
        return y_pred

    def _check_col_estimators(self, X, X_name="X", est_attr="estimators", cls=None):
        """Check getattr(self, est_attr) attribute, and coerce to (name, est, index).

        Checks:

        * `getattr(self, est_attr)` is single estimator, or
        * `getattr(self, est_attr)` is list of (name, estimator, index)
        * all `estimator` above inherit from `cls` (`None` means `BaseEstimator`)
        * `X.columns` is disjoint union of `index` appearing above

        Parameters
        ----------
        X : `pandas` object with `columns` attribute of `pd.Index` type
        X_name : str, optional, default = "X"
            name of `X` displayed in error messages
        est_attr : str, optional, default = "estimators"
            attribute name of the attribute this function checks
            also used in error message
        cls : type, optional, default = sktime BaseEstimator
            class to check inheritance from, for estimators (see above)

        Returns
        -------
        list of (name, estimator, index) such that union of index is `X.columns`;
        and estimator is estimator inheriting from `cls`

        Raises
        ------
        ValueError if checks fail, with informative error message
        """
        if cls is None:
            cls = BaseEstimator

        estimators = getattr(self, est_attr)

        # if a single estimator is passed, replicate across columns
        if isinstance(estimators, cls):
            ycols = [str(col) for col in X.columns]
            colrange = range(len(ycols))
            est_list = [estimators.clone() for _ in colrange]
            return list(zip(ycols, est_list, colrange))

        if (
            estimators is None
            or len(estimators) == 0
            or not isinstance(estimators, list)
        ):
            raise ValueError(
                f"Invalid '{est_attr}' attribute, '{est_attr}' should be a list"
                " of (string, estimator, int) tuples."
            )
        names, ests, indices = zip(*estimators)

        # check names, via _HeterogenousMetaEstimator._check_names
        if hasattr(self, "_check_names"):
            self._check_names(names)

        # coerce column names to indices in columns
        indices = self._get_indices(X, indices)

        for est in ests:
            if not isinstance(est, cls):
                raise ValueError(
                    f"The estimator {est.__class__.__name__} should be of type "
                    f"{cls}."
                )

        index_flat = flatten(indices)
        index_set = set(index_flat)
        not_in_y_idx = index_set.difference(X.columns)
        y_cols_not_found = set(X.columns).difference(index_set)

        if len(not_in_y_idx) > 0:
            raise ValueError(
                f"Column identifier must be indices in {X_name}.columns, or integers "
                f"within the range of the total number of columns, "
                f"but found column identifiers that are neither: {list(not_in_y_idx)}"
            )
        if len(y_cols_not_found) > 0:
            raise ValueError(
                f"All columns of {X_name} must be indexed by column identifiers, but "
                f"the following columns of {X_name} are not indexed: "
                f"{list(y_cols_not_found)}"
            )

        if len(index_set) != len(index_flat):
            raise ValueError(
                f"One estimator per column required. Found {len(index_set)} unique"
                f" column names in {est_attr} arg, required {len(index_flat)}"
            )

        return estimators
