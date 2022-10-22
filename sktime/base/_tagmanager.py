# -*- coding: utf-8 -*-
"""Mixin class for tag and configuration settings management."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["_TagManager"]


import inspect
from copy import deepcopy


class _TagManager:
    """Mixin class for tag and configuration settings management."""

    @classmethod
    def _get_class_tags(cls, tag_attr_name="_tags"):
        """Get class tags from estimator class and all its parent classes.

        Parameters
        ----------
        tag_attr_name : str, optional, default = "_tags"
            name of the tag attribute that is read

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
            if hasattr(parent_class, tag_attr_name):
                # Need the if here because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = getattr(parent_class, tag_attr_name)
                collected_tags.update(more_tags)

        return deepcopy(collected_tags)

    @classmethod
    def _get_class_tag(cls, tag_name, tag_value_default=None, tag_attr_name="_tags"):
        """Get tag value from estimator class (only class tags).

        Parameters
        ----------
        tag_name : str
            Name of tag value.
        tag_value_default : any type
            Default/fallback value if tag is not found.
        tag_attr_name : str, optional, default = "_tags"
            name of the tag attribute that is read

        Returns
        -------
        tag_value :
            Value of `tag_name` tag in self. If not found, returns `tag_value_default`.
        """
        collected_tags = cls._get_class_tags(tag_attr_name=tag_attr_name)

        return collected_tags.get(tag_name, tag_value_default)

    def _get_tags(self, tag_attr_name="_tags"):
        """Get tags from estimator class and dynamic tag overrides.

        Parameters
        ----------
        tag_attr_name : str, optional, default = "_tags"
            name of the tag attribute that is read

        Returns
        -------
        collected_tags : dict
            Dictionary of tag name : tag value pairs. Collected from tag_attr_name
            class attribute via nested inheritance and then any overrides
            and new tags from [tag_attr_name]_dynamic object attribute.
        """
        collected_tags = self._get_class_tags(tag_attr_name=tag_attr_name)

        if hasattr(self, f"{tag_attr_name}_dynamic"):
            collected_tags.update(getattr(self, f"{tag_attr_name}_dynamic"))

        return deepcopy(collected_tags)

    def _get_tag(
        self, tag_name, tag_value_default=None, raise_error=True, tag_attr_name="_tags"
    ):
        """Get tag value from estimator class and dynamic tag overrides.

        Parameters
        ----------
        tag_name : str
            Name of tag to be retrieved
        tag_value_default : any type, optional; default=None
            Default/fallback value if tag is not found
        raise_error : bool
            whether a ValueError is raised when the tag is not found
        tag_attr_name : str, optional, default = "_tags"
            name of the tag attribute that is read

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
        collected_tags = self._get_tags(tag_attr_name=tag_attr_name)

        tag_value = collected_tags.get(tag_name, tag_value_default)

        if raise_error and tag_name not in collected_tags.keys():
            raise ValueError(f"Tag with name {tag_name} could not be found.")

        return tag_value

    def _set_tags(self, tag_attr_name="_tags", **tag_dict):
        """Set dynamic tags to given values.

        Parameters
        ----------
        tag_dict : dict
            Dictionary of tag name : tag value pairs.
        tag_attr_name : str, optional, default = "_tags"
            name of the tag attribute that is read

        Returns
        -------
        Self : Reference to self.

        Notes
        -----
        Changes object state by settting tag values in tag_dict as dynamic tags
        in self.
        """
        tag_update = deepcopy(tag_dict)
        dynamic_tags = f"{tag_attr_name}_dynamic"
        if hasattr(self, dynamic_tags):
            getattr(self, dynamic_tags).update(tag_update)
        else:
            setattr(self, dynamic_tags, tag_update)

        return self

    def _clone_tags(self, estimator, tag_names=None, tag_attr_name="_tags"):
        """clone/mirror tags from another estimator as dynamic override.

        Parameters
        ----------
        estimator : estimator inheriting from :class:BaseEstimator
        tag_names : str or list of str, default = None
            Names of tags to clone. If None then all tags in estimator are used
            as `tag_names`.
        tag_attr_name : str, optional, default = "_tags"
            name of the tag attribute that is read

        Returns
        -------
        Self :
            Reference to self.

        Notes
        -----
        Changes object state by setting tag values in tag_set from estimator as
        dynamic tags in self.
        """
        tags_est = deepcopy(estimator._get_tags(tag_attr_name=tag_attr_name))

        # if tag_set is not passed, default is all tags in estimator
        if tag_names is None:
            tag_names = tags_est.keys()
        else:
            # if tag_set is passed, intersect keys with tags in estimator
            if not isinstance(tag_names, list):
                tag_names = [tag_names]
            tag_names = [key for key in tag_names if key in tags_est.keys()]

        update_dict = {key: tags_est[key] for key in tag_names}

        self._set_tags(tag_attr_name=tag_attr_name, **update_dict)

        return self
