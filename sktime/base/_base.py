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
    getting default parameters (all sets)         - get_test_params()
    get one test instance with default parameters - create_test_instance()
    get list of all test instances plus name list - create_test_instances_and_names()
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
import warnings
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

    @classmethod
    def create_test_instances_and_names(cls):
        """Create list of all test instances and a list of names for them.

        Returns
        -------
        objs : list of instances of cls
            i-th instance is cls(**cls.get_test_params()[i])
        names : list of str, same length as objs
            i-th element is name of i-th instance of obj in tests
            convention is {cls.__name__}-{i} if more than one instance
            otherwise {cls.__name__}
        """
        objs = []
        param_list = cls.get_test_params()
        if not isinstance(param_list, (dict, list)):
            raise RuntimeError(
                f"Error in {cls.__name__}.get_test_params, "
                "return must be param dict for class, or list thereof"
            )
        if isinstance(param_list, dict):
            param_list = [param_list]
        for params in param_list:
            if not isinstance(params, dict):
                raise RuntimeError(
                    f"Error in {cls.__name__}.get_test_params, "
                    "return must be param dict for class, or list thereof"
                )
            objs += [cls(**params)]

        num_instances = len(param_list)
        if num_instances > 1:
            names = [cls.__name__ + "-" + str(i) for i in range(num_instances)]
        else:
            names = [cls.__name__]

        return objs, names

    @classmethod
    def _has_implementation_of(cls, method):
        """Check if method has a concrete implementation in this class.

        This assumes that having an implementation is equivalent to
            one or more overrides of `method` in the method resolution order.

        Parameters
        ----------
        method : str, name of method to check implementation of

        Returns
        -------
        bool, whether method has implementation in cls
            True if cls.method has been overridden at least once in
                the inheritance tree (according to method resolution order)
        """
        # walk through method resolution order and inspect methods
        #   of classes and direct parents, "adjacent" classes in mro
        mro = inspect.getmro(cls)
        # collect all methods that are not none
        methods = [getattr(c, method, None) for c in mro]
        methods = [m for m in methods if m is not None]

        for i in range(len(methods) - 1):
            # the method has been overridden once iff
            #  at least two of the methods collected are not equal
            #  equivalently: some two adjacent methods are not equal
            overridden = methods[i] != methods[i + 1]
            if overridden:
                return True

        return False


class TagAliaserMixin:
    """Mixin class for tag aliasing and deprecation of old tags.

    To deprecate tags, add the TagAliaserMixin to BaseObject or BaseEstimator.
    alias_dict contains the deprecated tags, and supports removal and renaming.
        For removal, add an entry "old_tag_name": ""
        For renaming, add an entry "old_tag_name": "new_tag_name"
    deprecate_dict contains the version number of renaming or removal.
        the keys in deprecate_dict should be the same as in alias_dict.
        values in deprecate_dict should be strings, the version of removal/renaming.

    The class will ensure that new tags alias old tags and vice versa, during
    the deprecation period. Informative warnings will be raised whenever the
    deprecated tags are being accessed.

    When removing tags, ensure to remove the removed tags from this class.
    If no tags are deprecated anymore (e.g., all deprecated tags are removed/renamed),
    ensure toremove this class as a parent of BaseObject or BaseEstimator.
    """

    # dictionary of aliases
    # key = old tag; value = new tag, aliased by old tag
    alias_dict = {"fit-in-transform": "fit_is_empty", "fit-in-predict": "fit_is_empty"}

    # dictionary of removal version
    # key = old tag; value = version in which tag will be removed, as string
    deprecate_dict = {"fit-in-transform": "0.12.0", "fit-in-predict": "0.12.0"}

    def __init__(self):
        super(TagAliaserMixin, self).__init__()

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
        collected_tags = super(TagAliaserMixin, cls).get_class_tags()
        collected_tags = cls._complete_dict(collected_tags)
        return collected_tags

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
        cls._deprecate_tag_warn([tag_name])
        return super(TagAliaserMixin, cls).get_class_tag(
            tag_name=tag_name, tag_value_default=tag_value_default
        )

    def get_tags(self):
        """Get tags from estimator class and dynamic tag overrides.

        Returns
        -------
        collected_tags : dict
            Dictionary of tag name : tag value pairs. Collected from _tags
            class attribute via nested inheritance and then any overrides
            and new tags from _tags_dynamic object attribute.
        """
        collected_tags = super(TagAliaserMixin, self).get_tags()
        collected_tags = self._complete_dict(collected_tags)
        return collected_tags

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
        self._deprecate_tag_warn([tag_name])
        return super(TagAliaserMixin, self).get_tag(
            tag_name=tag_name,
            tag_value_default=tag_value_default,
            raise_error=raise_error,
        )

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
        self._deprecate_tag_warn(tag_dict.keys())

        tag_dict = self._complete_dict(tag_dict)
        super(TagAliaserMixin, self).set_tags(**tag_dict)
        return self

    @classmethod
    def _complete_dict(cls, tag_dict):
        """Add all aliased and aliasing tags to the dictionary."""
        alias_dict = cls.alias_dict
        deprecated_tags = set(tag_dict.keys()).intersection(alias_dict.keys())
        new_tags = set(tag_dict.keys()).intersection(alias_dict.values())

        if len(deprecated_tags) > 0 or len(new_tags) > 0:
            new_tag_dict = deepcopy(tag_dict)
            # for all tag strings being set, write the value
            #   to all tags that could *be aliased by* the string
            #   and all tags that could be *aliasing* the string
            # this way we ensure upwards and downwards compatibility
            for old_tag, new_tag in alias_dict.items():
                for tag in tag_dict:
                    if tag == old_tag and new_tag != "":
                        new_tag_dict[new_tag] = tag_dict[tag]
                    if tag == new_tag:
                        new_tag_dict[old_tag] = tag_dict[tag]
            return new_tag_dict
        else:
            return tag_dict

    @classmethod
    def _deprecate_tag_warn(cls, tags):
        """Print warning message for tag deprecation.

        Parameters
        ----------
        tags : list of str

        Raises
        ------
        DeprecationWarning for each tag in tags that is aliased by cls.alias_dict
        """
        for tag_name in tags:
            if tag_name in cls.alias_dict.keys():
                version = cls.deprecate_dict[tag_name]
                new_tag = cls.alias_dict[tag_name]
                msg = f'tag "{tag_name}" is will be removed in sktime version {version}'
                if new_tag != "":
                    msg += (
                        f' and replaced by "{new_tag}", please use "{new_tag}" instead'
                    )
                else:
                    msg += ', please remove code that access or sets "{tag_name}"'
                warnings.warn(msg, category=DeprecationWarning)


class BaseEstimator(TagAliaserMixin, BaseObject):
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
