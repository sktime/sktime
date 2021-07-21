# -*- coding: utf-8 -*-
"""
Base class template for objects and fittable objects.

templates in this module:

    BaseObject - object with parameters and tags
    BaseEstimator - BaseObject that can be fitted

Interface specifications below.

---

    class name: BaseObject

Hyper-parameter inspection and setter methods:
    inspect hyper-parameters      - get_params()
    setting hyper-parameters      - set_params(**params)

Tag inspection and setter methods
    inspect tags (all)            - get_tags()
    inspect tags (one tag)        - get_tag(tag_name: str, tag_value_default=None)
    inspect tags (class method)   - get_class_tags()
    inspect tags (one tag, class) - get_class_tag(tag_name:str, tag_value_default=None)
    setting dynamic tags          - set_tag(**tag_dict: dict)
    set/clone dynamic tags        - clone_tags(estimator, tag_names=None)

---

    class name: BaseEstimator

Provides all interface points of BaseObject, plus:

Parameter inspection:
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state check      - check_is_fitted (raises error if not is_fitted)

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["mloning", "RNKuhns", "fkiraly"]
__all__ = ["BaseEstimator", "BaseObject"]

import inspect

from copy import deepcopy

from sklearn import clone
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.ensemble._base import _set_random_states

from sktime.exceptions import NotFittedError


class BaseObject(_BaseEstimator):
    """Base class for parametric objects with tags sktime.

    Extends scikit-learn's BaseEstimator to include sktime interface for tags.
    """

    def __init__(self):
        self._tags_dynamic = dict()
        super(BaseObject, self).__init__()

    @classmethod
    def get_class_tags(cls):
        """Get class tags from estimator class and all its parent classes.

        Returns
        -------
        collected_tags : dictionary of tag names : tag values
            collected from _tags class attribute via nested inheritance
            NOT overridden by dynamic tags set by set_tags or mirror_tags
        """
        collected_tags = dict()

        # We exclude the last two parent classes: sklearn.base.BaseEstimator and
        # the basic Python object.
        for parent_class in reversed(inspect.getmro(cls)[:-2]):
            if hasattr(parent_class, "_tags"):
                # Need the if here because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = parent_class._tags
                collected_tags.update(more_tags)

        return deepcopy(collected_tags)

    @classmethod
    def get_class_tag(cls, tag_name, tag_value_default=None):
        """Get tag value from estimator class (only class tags).

        Parameters
        ----------
        tag_name : str, name of tag value
        tag_value_default : any type, default/fallback value if tag is not found

        Returns
        -------
        tag_value : value of the tag tag_name in self if found
                    if tag is not found, returns tag_value_default
        """
        collected_tags = cls.get_class_tags()

        return collected_tags.get(tag_name, tag_value_default)

    def get_tags(self):
        """Get tags from estimator class and dynamic tag overrides.

        Returns
        -------
        collected_tags : dictionary of tag names : tag values
            collected from _tags class attribute via nested inheritance
            then any overrides and new tags from _tags_dynamic object attribute
        """
        collected_tags = self.get_class_tags()

        if hasattr(self, "_tags_dynamic"):
            collected_tags.update(self._tags_dynamic)

        return deepcopy(collected_tags)

    def get_tag(self, tag_name, tag_value_default=None):
        """Get tag value from estimator class and dynamic tag overrides.

        Parameters
        ----------
        tag_name : str, name of tag value
        tag_value_default : any type, default/fallback value if tag is not found

        Returns
        -------
        tag_value : value of the tag tag_name in self if found
                    if tag is not found, returns tag_value_default
        """
        collected_tags = self.get_tags()

        return collected_tags.get(tag_name, tag_value_default)

    def set_tags(self, **tag_dict):
        """Set dynamic tags to given values.

        Parameters
        ----------
        tag_dict : dictionary of tag names : tag values

        Returns
        -------
        reference to self

        State change
        ------------
        sets tag values in tag_dict as dynamic tags in self
        """
        self._tags_dynamic.update(deepcopy(tag_dict))

        return self

    def clone_tags(self, estimator, tag_names=None):
        """clone/mirror tags from another estimator as dynamic override.

        Parameters
        ----------
        estimator : an estimator inheriting from BaseEstimator
        tag_names : list of str, or str; names of tags to clone
            default = list of all tags in estimator

        Returns
        -------
        reference to self

        State change
        ------------
        sets tag values in tag_set from estimator as dynamic tags in self
        """
        tags_est = deepcopy(estimator.get_tags())

        # if tag_set is not passed, default is all tags in estimator
        if tag_names is None:
            tag_names = tags_est.keys()
        else:
            # if tag_set is passed, intersect keys with tags in estimator
            if not isinstance(tag_names, list):
                tag_names = [tag_names]
            tag_names = [key for key in tag_names if key in tags_est.keys()]

        update_dict = {key: tags_est[key] for key in tag_names}

        self.set_tags(**update_dict)

        return self


class BaseEstimator(BaseObject):
    """Base class for defining estimators in sktime.

    Extends sktime's BaseObject to include basic functionality for fittable estimators.
    """

    def __init__(self):
        self._is_fitted = False
        super(BaseEstimator, self).__init__()

    @property
    def is_fitted(self):
        """Whether `fit` has been called."""
        return self._is_fitted

    def check_is_fitted(self):
        """Check if the estimator has been fitted.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )


def _clone_estimator(base_estimator, random_state=None):
    estimator = clone(base_estimator)

    if random_state is not None:
        _set_random_states(estimator, random_state)

    return estimator
