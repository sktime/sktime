#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements meta estimator for estimators composed of other estimators."""

__author__ = ["mloning, fkiraly"]
__all__ = ["_HeterogenousMetaEstimator"]

from abc import ABCMeta

from sklearn import clone

from sktime.base import BaseEstimator


class _HeterogenousMetaEstimator(BaseEstimator, metaclass=ABCMeta):
    """Handles parameter management for estimators composed of named estimators.

    Partly adapted from sklearn utils.metaestimator.py.
    """

    def get_params(self, deep=True):
        """Return estimator parameters."""
        raise NotImplementedError("abstract method")

    def set_params(self, **params):
        """Set estimator parameters."""
        raise NotImplementedError("abstract method")

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

    def _subset_dict_keys(self, dict_to_subset, keys):
        """Subset dictionary d to keys in keys."""
        keys_in_both = set(keys).intersection(dict_to_subset.keys())
        subsetted_dict = dict((k, dict_to_subset[k]) for k in keys_in_both)
        return subsetted_dict

    def _check_estimators(self, estimators, attr_name="steps", cls_type=None):
        """Check that estimators is a list of estimators or list of str/est tuples.

        Parameters
        ----------
        estimators : any object
            should be list of estimators or list of (str, estimator) tuples
            estimators should inherit from cls_type class
        attr_name : str, optional. Default = "steps"
            Name of checked attribute in error messages
        cls_type : class, optional. Default = BaseEstimator.
            class that all estimators are checked to be an instance of

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
            cls_type = BaseEstimator
        else:
            msg += f"All estimators must be of type {cls_type}."

        if (
            estimators is None
            or len(estimators) == 0
            or not isinstance(estimators, list)
        ):
            raise TypeError(msg)

        if not isinstance(estimators[0], (cls_type, tuple)):
            raise TypeError(msg)

        if isinstance(estimators[0], cls_type):
            if not all(isinstance(est, cls_type) for est in estimators):
                raise TypeError(msg)
        if isinstance(estimators[0], tuple):
            if not all(isinstance(est, tuple) for est in estimators):
                raise TypeError(msg)
            if not all(isinstance(est[0], str) for est in estimators):
                raise TypeError(msg)
            if not all(isinstance(est[1], cls_type) for est in estimators):
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
        elif isinstance(estimators[0], BaseEstimator):
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
            if was a list of estimators, then str are generated via _get_estimator_names
        """
        ests = self._get_estimator_list(estimators)
        if clone_ests:
            ests = [clone(e) for e in ests]
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
        tag with name tag_name, sets to value if _anytagis(tag_name, value) is True
            otherwise sets the tag to `value_if_not`

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
