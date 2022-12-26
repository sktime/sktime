#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements meta estimator for estimators composed of other estimators."""

__author__ = ["mloning, fkiraly"]
__all__ = ["_HeterogenousMetaEstimator"]

from inspect import isclass

from sktime.base import BaseEstimator


class _HeterogenousMetaEstimator:
    """Handles parameter management for estimators composed of named estimators.

    Partly adapted from sklearn utils.metaestimator.py.
    """

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator) pairs for the default
    _steps_attr = "_steps"

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
        steps = self._steps_attr
        return self._get_params(steps, deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of estimator in `_forecasters`.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self : returns an instance of self.
        """
        steps_attr = self._steps_attr
        self._set_params(steps_attr, **kwargs)
        return self

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

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, "get_params"):
                for key, value in estimator.get_params(deep=True).items():
                    out["%s__%s" % (name, key)] = value
        return out

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        items = getattr(self, attr)
        names = []
        if items:
            names, _ = zip(*items)
        for name in list(params.keys()):
            if "__" not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _check_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: {0!r}".format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                "Estimator names conflict with constructor "
                "arguments: {0!r}".format(sorted(invalid_names))
            )
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                "Estimator names must not contain __: got "
                "{0!r}".format(invalid_names)
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
        subsetted_dict = dict((rem_prefix(k), dict_to_subset[k]) for k in keys_in_both)
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
        bool : True if obj is (str, cls_type) tuple, False otherise
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
            f"Invalid '{attr_name}' attribute, '{attr_name}' should be a list"
            " of estimators, or a list of (string, estimator) tuples. "
        )
        if cls_type is None:
            msg += f"All estimators in '{attr_name}' must be of type BaseEstimator."
            cls_type = BaseEstimator
        elif isclass(cls_type) or isinstance(cls_type, tuple):
            msg += (
                f"All estimators in '{attr_name}' must be of type "
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
        self, other, base_class, composite_class, attr_name="steps", concat_order="left"
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
                f'"{concat_order}"'
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

        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_ests)):
            return composite_class(**{attr_name: list(new_ests)})
        else:
            return composite_class(**{attr_name: list(zip(new_names, new_ests))})

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
