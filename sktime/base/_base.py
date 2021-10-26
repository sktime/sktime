# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Base class template for objects and fittable objects.

templates in this module:

    BaseObject - object with parameters and tags
    BaseEstimator - BaseObject that can be fitted

Interface specifications below.

---

    class name: BaseObject

Hyper-parameter inspection and setter methods:
    inspect hyper-parameters     - get_params()
    setting hyper-parameters     - set_params(**params)

Tag inspection and setter methods
    inspect tags (all)            - get_tags()
    inspect tags (one tag)        - get_tag(tag_name: str, tag_value_default=None)
    inspect tags (class method)   - get_class_tags()
    inspect tags (one tag, class) - get_class_tag(tag_name:str, tag_value_default=None)
    setting dynamic tags          - set_tag(**tag_dict: dict)
    set/clone dynamic tags        - clone_tags(estimator, tag_names=None)

Testing with default parameters methods
    getting default parameters           - get_test_params()
    get instance with default parameters - create_test_instance()

---

    class name: BaseEstimator

Provides all interface points of BaseObject, plus:

Parameter inspection:
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state check      - check_is_fitted (raises error if not is_fitted)
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
        collected_tags : dict
            Dictionary of tag name : tag value pairs. Collected from _tags
            class attribute via nested inheritance. NOT overridden by dynamic
            tags set by set_tags or mirror_tags.
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
        tag_name : str
            Name of tag value.
        tag_value_default : any type
            Default/fallback value if tag is not found.

        Returns
        -------
        tag_value :
            Value of the `tag_name` tag in self. If not found, returns
            `tag_value_default`.
        """
        collected_tags = cls.get_class_tags()

        return collected_tags.get(tag_name, tag_value_default)

    def get_tags(self):
        """Get tags from estimator class and dynamic tag overrides.

        Returns
        -------
        collected_tags : dict
            Dictionary of tag name : tag value pairs. Collected from _tags
            class attribute via nested inheritance and then any overrides
            and new tags from _tags_dynamic object attribute.
        """
        collected_tags = self.get_class_tags()

        if hasattr(self, "_tags_dynamic"):
            collected_tags.update(self._tags_dynamic)

        return deepcopy(collected_tags)

    def get_tag(self, tag_name, tag_value_default=None, raise_error=True):
        """Get tag value from estimator class and dynamic tag overrides.

        Parameters
        ----------
        tag_name : str
            Name of tag to be retrieved
        tag_value_default : any type, optional; default=None
            Default/fallback value if tag is not found
        raise_error : bool
            whether a ValueError is raised when the tag is not found

        Returns
        -------
        tag_value :
            Value of the `tag_name` tag in self. If not found, returns an error if
            raise_error is True, otherwise it returns `tag_value_default`.

        Raises
        ------
        ValueError if raise_error is True i.e. if tag_name is not in self.get_tags(
        ).keys()
        """
        collected_tags = self.get_tags()

        tag_value = collected_tags.get(tag_name, tag_value_default)

        if raise_error and tag_name not in collected_tags.keys():
            raise ValueError(f"Tag with name {tag_name} could not be found.")

        return tag_value

    def set_tags(self, **tag_dict):
        """Set dynamic tags to given values.

        Parameters
        ----------
        tag_dict : dict
            Dictionary of tag name : tag value pairs.

        Returns
        -------
        Self :
            Reference to self.

        Notes
        -----
        Changes object state by settting tag values in tag_dict as dynamic tags
        in self.
        """
        self._tags_dynamic.update(deepcopy(tag_dict))

        return self

    def clone_tags(self, estimator, tag_names=None):
        """clone/mirror tags from another estimator as dynamic override.

        Parameters
        ----------
        estimator : estimator inheriting from :class:BaseEstimator
        tag_names : str or list of str, default = None
            Names of tags to clone. If None then all tags in estimator are used
            as `tag_names`.

        Returns
        -------
        Self :
            Reference to self.

        Notes
        -----
        Changes object state by setting tag values in tag_set from estimator as
        dynamic tags in self.
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

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # imported inside the function to avoid circular imports
        from sktime.tests._config import ESTIMATOR_TEST_PARAMS

        # if non-default parameters are required, but none have been found,
        # raise error
        if hasattr(cls, "_required_parameters"):
            required_parameters = getattr(cls, "required_parameters", [])
            if len(required_parameters) > 0:
                raise ValueError(
                    f"Estimator: {cls} requires "
                    f"non-default parameters for construction, "
                    f"but none were given. Please set them "
                    f"as given in the extension template"
                )

        # construct with parameter configuration for testing, otherwise construct with
        # default parameters (empty dict)
        params = ESTIMATOR_TEST_PARAMS.get(cls, {})
        return params

    @classmethod
    def create_test_instance(cls):
        """Construct Estimator instance if possible.

        Returns
        -------
        instance : instance of the class with default parameters

        Notes
        -----
        `get_test_params` can return dict or list of dict.
        This function takes first or single dict that get_test_params returns, and
        constructs the object with that.
        """
        params = cls.get_test_params()
        if isinstance(params, list):
            if isinstance(params[0], dict):
                params = params[0]
            else:
                raise TypeError(
                    "get_test_params should either return a dict or list of dict."
                )
        elif isinstance(params, dict):
            pass
        else:
            raise TypeError(
                "get_test_params should either return a dict or list of dict."
            )

        return cls(**params)


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
