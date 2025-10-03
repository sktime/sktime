# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for objects and fittable objects.

templates in this module:

    BaseObject - object with parameters and tags
    BaseEstimator - BaseObject that can be fitted

Interface specifications below.

---

    class name: BaseObject

Parameter inspection and setter methods
    inspect parameter values      - get_params()
    setting parameter values      - set_params(**params)
    list of parameter names       - get_param_names()
    dict of parameter defaults    - get_param_defaults()

Tag inspection and setter methods
    inspect tags (all)            - get_tags()
    inspect tags (one tag)        - get_tag(tag_name: str, tag_value_default=None)
    inspect tags (class method)   - get_class_tags()
    inspect tags (one tag, class) - get_class_tag(tag_name:str, tag_value_default=None)
    setting dynamic tags          - set_tags(**tag_dict: dict)
    set/clone dynamic tags        - clone_tags(estimator, tag_names=None)

Config inspection and setter methods
    get configs (all)             - get_config()
    set configs                   - set_config(**config_dict: dict)

Blueprinting: resetting and cloning, post-init state with same hyper-parameters
    reset estimator to post-init  - reset()
    cloneestimator (copy&reset)   - clone()

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

from copy import deepcopy

from skbase.base import BaseEstimator as _BaseEstimator
from skbase.base import BaseObject as _BaseObject
from skbase.base._base import TagAliaserMixin as _TagAliaserMixin
from sklearn import clone
from sklearn.base import BaseEstimator as _SklearnBaseEstimator

from sktime import __version__ as SKTIME_VERSION
from sktime.utils._estimator_html_repr import _HTMLDocumentationLinkMixin
from sktime.utils.random_state import set_random_state

SERIALIZATION_FORMATS = {
    "pickle",
    "cloudpickle",
}


class BaseObject(_HTMLDocumentationLinkMixin, _BaseObject):
    """Base class for parametric objects with tags in sktime.

    Base class for all parametric objects in sktime.

    Extends skbase BaseObject with additional features.
    """

    # global default tags for dependency management
    _tags = {
        "python_version": None,  # PEP 440 version specifier, e.g., ">=3.7"
        "python_dependencies": None,  # PEP 440 dependency strs, e.g., "pandas>=1.0"
        "env_marker": None,  # PEP 508 environment marker, e.g., "os_name=='posix'"
        "sktime_version": SKTIME_VERSION,  # current sktime version
        # default property tags
        "property:randomness": "deterministic",
        "capability:random_state": False,
        # default tags for testing
        "tests:core": False,  # core objects have wider trigger conditions in testing
        "tests:vm": False,  # whether the object should be tested in its own VM
        "tests:libs": None,  # required libraries, for change conditional testing
        "tests:skip_all": False,  # whether all tests for the object should be skipped
        "tests:skip_by_name": None,  # list of test names to skip for this object
    }

    _config = {
        "warnings": "on",
        "backend:parallel": None,  # parallelization backend for broadcasting
        #  {None, "dask", "loky", "multiprocessing", "threading","ray"}
        #  None: no parallelization
        #  "loky", "multiprocessing" and "threading": uses `joblib` Parallel loops
        #  "dask": uses `dask`, requires `dask` package in environment
        #  "ray": uses `ray`, requires `ray` package in environment
        "backend:parallel:params": None,  # params for parallelization backend,
    }

    _config_doc = {
        "display": """
        display : str, "diagram" (default), or "text"
            how jupyter kernels display instances of self

            * "diagram" = html box diagram representation
            * "text" = string printout
        """,
        "print_changed_only": """
        print_changed_only : bool, default=True
            whether printing of self lists only self-parameters that differ
            from defaults (False), or all parameter names and values (False).
            Does not nest, i.e., only affects self and not component estimators.
        """,
        "warnings": """
        warnings : str, "on" (default), or "off"
            whether to raise warnings, affects warnings from sktime only

            * "on" = will raise warnings from sktime
            * "off" = will not raise warnings from sktime
        """,
        "backend:parallel": """
        backend:parallel : str, optional, default="None"
            backend to use for parallelization when broadcasting/vectorizing, one of

            - "None": executes loop sequentially, simple list comprehension
            - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel``
            - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
            - "dask": uses ``dask``, requires ``dask`` package in environment
            - "ray": uses ``ray``, requires ``ray`` package in environment
        """,
        "backend:parallel:params": """
        backend:parallel:params : dict, optional, default={} (no parameters passed)
            additional parameters passed to the parallelization backend as config.
            Valid keys depend on the value of ``backend:parallel``:

            - "None": no additional parameters, ``backend_params`` is ignored

            - "loky", "multiprocessing" and "threading": default ``joblib`` backends
              any valid keys for ``joblib.Parallel`` can be passed here, e.g.,
              ``n_jobs``, with the exception of ``backend`` which is directly
              controlled by ``backend``.
              If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
              will default to ``joblib`` defaults.

            - "joblib": custom and 3rd party ``joblib`` backends,
              e.g., ``spark``. Any valid keys for ``joblib.Parallel``
              can be passed here, e.g., ``n_jobs``,
              ``backend`` must be passed as a key of ``backend_params`` in this case.
              If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
              will default to ``joblib`` defaults.

            - "dask": any valid keys for ``dask.compute`` can be passed,
              e.g., ``scheduler``

            - "ray": The following keys can be passed:

                - "ray_remote_args": dictionary of valid keys for ``ray.init``
                - "shutdown_ray": bool, default=True; False prevents ``ray`` from
                    shutting down after parallelization.
                - "logger_name": str, default="ray"; name of the logger to use.
                - "mute_warnings": bool, default=False; if True, suppresses warnings
        """,
    }

    def __eq__(self, other):
        """Equality dunder. Checks equal class and parameters.

        Returns True iff result of get_params(deep=False) results in equal parameter
        sets.

        Nested BaseObject descendants from get_params are compared via __eq__ as well.
        """
        from sktime.utils.deep_equals import deep_equals

        if not isinstance(other, BaseObject):
            return False

        self_params = self.get_params(deep=False)
        other_params = other.get_params(deep=False)

        return deep_equals(self_params, other_params)

    @classmethod
    def _get_set_config_doc(cls):
        """Create docstring for set_config from self._config_doc.

        Returns
        -------
        collected_config_docs : dict
            Dictionary of doc name: docstring part.
            Collected from _config_doc class attribute via nested inheritance.
        """
        cfgs_dict = cls._get_class_flags(flag_attr_name="_config_doc")

        doc_start = """Set config flags to given values.

        Parameters
        ----------
        config_dict : dict
            Dictionary of config name : config value pairs.
            Valid configs, values, and their meaning is listed below:
        """

        doc_end = """
        Returns
        -------
        self : reference to self.

        Notes
        -----
        Changes object state, copies configs in config_dict to self._config_dynamic.
        """

        doc = doc_start
        for _, cfg_doc in cfgs_dict.items():
            doc += cfg_doc
        doc += doc_end
        return doc

    @classmethod
    def _init_dynamic_doc(cls):
        """Set docstring for set_config from self._config_doc."""
        try:  # try/except to avoid unexpected failures
            cls.set_config = deepcopy_func(cls.set_config)
            cls.set_config.__doc__ = cls._get_set_config_doc()
        except Exception:  # noqa: S110
            pass

    def save(self, path=None, serialization_format="pickle"):
        """Save serialized self to bytes-like object or to (.zip) file.

        Behaviour:

        * if ``path`` is None, returns an in-memory serialized self
        * if ``path`` is a file location, stores self at that location as a zip file

        saved files are zip files with following contents:

        * ``_metadata`` - contains class of self, i.e., ``type(self)``
        * ``_obj`` - serialized self. This class uses the default serialization
          (pickle).

        Parameters
        ----------
        path : None or file location (str or Path)
            if None, self is saved to an in-memory object
            if file location, self is saved to that file location. If:

            - path="estimator" then a zip file ``estimator.zip`` will be made at cwd.
            - path="/home/stored/estimator" then a zip file ``estimator.zip`` will be
            stored in ``/home/stored/``.

        serialization_format: str, default = "pickle"
            Module to use for serialization.
            The available options are "pickle" and "cloudpickle".
            Note that non-default formats might require
            installation of other soft dependencies.

        Returns
        -------
        if ``path`` is None - in-memory serialized self
        if ``path`` is file location - ZipFile with reference to the file
        """
        import pickle
        import shutil
        from pathlib import Path
        from zipfile import ZipFile

        from sktime.utils.dependencies import _check_soft_dependencies

        if serialization_format not in SERIALIZATION_FORMATS:
            raise ValueError(
                f"The provided `serialization_format`='{serialization_format}' "
                "is not yet supported. The possible formats are: "
                f"{SERIALIZATION_FORMATS}."
            )

        if path is not None and not isinstance(path, (str, Path)):
            raise TypeError(
                "`path` is expected to either be a string or a Path object "
                f"but found of type:{type(path)}."
            )
        if path is not None:
            path = Path(path) if isinstance(path, str) else path
            path.mkdir()

        if serialization_format == "cloudpickle":
            _check_soft_dependencies("cloudpickle", severity="error")
            import cloudpickle

            if path is None:
                return (type(self), cloudpickle.dumps(self))

            with open(path / "_metadata", "wb") as file:
                cloudpickle.dump(type(self), file)
            with open(path / "_obj", "wb") as file:
                cloudpickle.dump(self, file)

        elif serialization_format == "pickle":
            if path is None:
                return (type(self), pickle.dumps(self))

            with open(path / "_metadata", "wb") as file:
                pickle.dump(type(self), file)
            with open(path / "_obj", "wb") as file:
                pickle.dump(self, file)

        shutil.make_archive(base_name=path, format="zip", root_dir=path)
        shutil.rmtree(path)
        return ZipFile(path.with_name(f"{path.stem}.zip"))

    @classmethod
    def load_from_serial(cls, serial):
        """Load object from serialized memory container.

        Parameters
        ----------
        serial : 1st element of output of ``cls.save(None)``

        Returns
        -------
        deserialized self resulting in output ``serial``, of ``cls.save(None)``
        """
        import pickle

        return pickle.loads(serial)

    @classmethod
    def load_from_path(cls, serial):
        """Load object from file location.

        Parameters
        ----------
        serial : result of ZipFile(path).open("object)

        Returns
        -------
        deserialized self resulting in output at ``path``, of ``cls.save(path)``
        """
        import pickle
        from zipfile import ZipFile

        with ZipFile(serial, "r") as file:
            return pickle.loads(file.open("_obj").read())


class TagAliaserMixin(_TagAliaserMixin):
    """Mixin class for tag aliasing and deprecation of old tags.

    To deprecate tags, add the ``TagAliaserMixin`` to ``BaseObject``
    or ``BaseEstimator``.

    ``alias_dict`` contains the deprecated tags, and supports removal and renaming.

    * For removal, add an entry ``"old_tag_name": ""``
    * For renaming, add an entry ``"old_tag_name": "new_tag_name"``

    ``deprecate_dict`` contains the version number of renaming or removal.

    * The keys in ``deprecate_dict`` should be the same as in alias_dict.
    * Values in ``deprecate_dict`` should be strings, the version of
    removal/renaming, in PEP 440 format, e.g., ``"1.0.0"``.

    The class will ensure that new tags alias old tags and vice versa, during the
    deprecation period. Informative warnings will be raised whenever the deprecated tags
    are being accessed.

    When removing tags, ensure to remove the removed tags from this class. If no tags
    are deprecated anymore (e.g., all deprecated tags are removed/renamed), ensure
    to remove this class as a parent of ``BaseObject`` or ``BaseEstimator``.

    Exact aliasing logic, in the situation of an "old" and "new" tag,
    i.e., an entry ``{"old_tag": "new_tag"}`` in ``alias_dict``:

    * if only the new tag is present, and the new tag is requested,
      returns the value of the new tag, no warning.
      This is the "target" state of the deprecation process.
    * if both new and old tag are present, and any of the two is requested,
      returns the value of the old tag.
      This priority is in order to deprecate in a way that does not break existing code,
      in case the values of the two tags differ.
      Raises a warning in addition, if the old tag was requested.
    * if only the new tag is present, and the old tag is requested,
      returns the value of the new tag, raises a warning.
    * if only the old tag is present, and the new tag is requested,
      returns the value of the new tag, without a warning.

    Note: all warnings above are for the user of the estimator,
    when attempting to read the old tag,
    suggesting to use the new tag instead, which is the deprecation target state.

    Warnings and errors for the developer of the estimator,
    to change the old tag to new if
    the old tag is still present, are not raised by this class.
    These warnings should be raised separately, in API conformance tests,
    preferably at CI time and as exceptions.
    """

    alias_dict = {
        "handles-missing-data": "capability:missing_values",
        "ignores-exogeneous-X": "capability:exogenous",
        "univariate-only": "capability:multivariate",
    }
    deprecate_dict = {
        "handles-missing-data": "1.0.0",
        "ignores-exogeneous-X": "1.0.0",
        "univariate-only": "1.0.0",
    }

    @classmethod
    def get_class_tag(cls, tag_name, tag_value_default=None):
        """Get class tag value from class, with tag level inheritance from parents.

        Every ``scikit-base`` compatible object has a dictionary of tags,
        which are used to store metadata about the object.

        The ``get_class_tag`` method is a class method,
        and retrieves the value of a tag
        taking into account only class-level tag values and overrides.

        It returns the value of the tag with name ``tag_name`` from the object,
        taking into account tag overrides, in the following
        order of descending priority:

        1. Tags set in the ``_tags`` attribute of the class.
        2. Tags set in the ``_tags`` attribute of parent classes,
          in order of inheritance.

        Does not take into account dynamic tag overrides on instances,
        set via ``set_tags`` or ``clone_tags``,
        that are defined on instances.

        To retrieve tag values with potential instance overrides, use
        the ``get_tag`` method instead.

        Parameters
        ----------
        tag_name : str
            Name of tag value.
        tag_value_default : any type
            Default/fallback value if tag is not found.

        Returns
        -------
        tag_value :
            Value of the ``tag_name`` tag in ``self``.
            If not found, returns ``tag_value_default``.
        """
        cls._deprecate_tag_warn([tag_name])
        alias_dict = cls.alias_dict

        # check is tag is aliased or aliasing
        # if yes, ensure that tag_name is the new tag name str
        # and old_tag is the old tag name str
        old_tag_name = ""
        new_tag_name = ""
        if tag_name in alias_dict:
            old_tag_name = tag_name
            new_tag_name = alias_dict[old_tag_name]
        if tag_name in alias_dict.values():
            old_tag_name = [k for k, v in alias_dict.items() if v == tag_name][0]
            new_tag_name = tag_name

        tag_changed = new_tag_name != old_tag_name
        new_tag_queried = tag_name == new_tag_name
        old_tag_queried = tag_name == old_tag_name and tag_changed

        if tag_changed:
            # retrieve old tag value, if it exists
            old_tag_val = cls._get_class_flag(
                old_tag_name,
                "__tag_not_found__",
                flag_attr_name="_tags",
            )
            old_tag_present = old_tag_val != "__tag_not_found__"
            # case 1: old tag present, and new or old tag queried
            # then: return value of old tag
            if old_tag_present:
                # negate if new tag was queried and tag is in FLIPPED_TAGS
                # todo 1.0.0 - remove this special case
                if new_tag_queried and old_tag_name in cls.FLIPPED_TAGS:
                    return not old_tag_val
                return old_tag_val
            # case 2: old tag was queried, but old tag not present
            # then: return value of new tag
            # negate if tag is in FLIPPED_TAGS
            # todo 1.0.0 - remove this special case
            elif old_tag_queried:
                new_tag_value = cls._get_class_flag(
                    new_tag_name,
                    tag_value_default,
                    flag_attr_name="_tags",
                )
                if old_tag_queried and old_tag_name in cls.FLIPPED_TAGS:
                    return not new_tag_value
                return new_tag_value

        # if we reach here, then:
        # no aliasing happened, i.e., tag_name is not in alias_dict
        # then: return value of tag_name as usual
        tag_val = super().get_class_tag(
            tag_name=tag_name, tag_value_default=tag_value_default
        )
        return tag_val

    def get_tag(self, tag_name, tag_value_default=None, raise_error=True):
        """Get tag value from instance, with tag level inheritance and overrides.

        Every ``scikit-base`` compatible object has a dictionary of tags.
        Tags may be used to store metadata about the object,
        or to control behaviour of the object.

        Tags are key-value pairs specific to an instance ``self``,
        they are static flags that are not changed after construction
        of the object.

        The ``get_tag`` method retrieves the value of a single tag
        with name ``tag_name`` from the instance,
        taking into account tag overrides, in the following
        order of descending priority:

        1. Tags set via ``set_tags`` or ``clone_tags`` on the instance,
          at construction of the instance.
        2. Tags set in the ``_tags`` attribute of the class.
        3. Tags set in the ``_tags`` attribute of parent classes,
          in order of inheritance.

        Parameters
        ----------
        tag_name : str
            Name of tag to be retrieved
        tag_value_default : any type, optional; default=None
            Default/fallback value if tag is not found
        raise_error : bool
            whether a ``ValueError`` is raised when the tag is not found

        Returns
        -------
        tag_value : Any
            Value of the ``tag_name`` tag in ``self``.
            If not found, raises an error if
            ``raise_error`` is True, otherwise it returns ``tag_value_default``.

        Raises
        ------
        ValueError, if ``raise_error`` is ``True``.
            The ``ValueError`` is then raised if ``tag_name`` is
            not in ``self.get_tags().keys()``.
        """
        self._deprecate_tag_warn([tag_name])
        alias_dict = self.alias_dict

        old_tag_name = ""
        new_tag_name = ""
        if tag_name in alias_dict:
            old_tag_name = tag_name
            new_tag_name = alias_dict[old_tag_name]
        if tag_name in alias_dict.values():
            old_tag_name = [k for k, v in alias_dict.items() if v == tag_name][0]
            new_tag_name = tag_name

        tag_changed = new_tag_name != old_tag_name
        new_tag_queried = tag_name == new_tag_name
        old_tag_queried = tag_name == old_tag_name and tag_changed

        if tag_changed:
            # retrieve old tag value, if it exists
            old_tag_val = self._get_flag(
                old_tag_name,
                "__tag_not_found__",
                raise_error=False,
                flag_attr_name="_tags",
            )
            old_tag_present = old_tag_val != "__tag_not_found__"
            # case 1: old tag present, and new or old tag queried
            # then: return value of old tag
            if old_tag_present:
                # negate if new tag was queried and tag is in FLIPPED_TAGS
                # todo 1.0.0 - remove this special case
                if new_tag_queried and old_tag_name in self.FLIPPED_TAGS:
                    return not old_tag_val
                return old_tag_val
            # case 2: old tag was queried, but old tag not present
            # then: return value of new tag
            # negate if tag is in FLIPPED_TAGS
            # todo 1.0.0 - remove this special case
            elif old_tag_queried:
                new_tag_value = self._get_flag(
                    new_tag_name,
                    tag_value_default,
                    raise_error=False,
                    flag_attr_name="_tags",
                )
                if old_tag_queried and old_tag_name in self.FLIPPED_TAGS:
                    return not new_tag_value
                return new_tag_value

        # if we reach here, then:
        # no aliasing happened, i.e., tag_name is not in alias_dict
        # then: return value of tag_name as usual
        tag_val = super().get_tag(
            tag_name=tag_name,
            tag_value_default=tag_value_default,
            raise_error=raise_error,
        )
        return tag_val

    def set_tags(self, **tag_dict):
        """Set instance level tag overrides to given values.

        Every ``scikit-base`` compatible object has a dictionary of tags,
        which are used to store metadata about the object.

        Tags are key-value pairs specific to an instance ``self``,
        they are static flags that are not changed after construction
        of the object. They may be used for metadata inspection,
        or for controlling behaviour of the object.

        ``set_tags`` sets dynamic tag overrides
        to the values as specified in ``tag_dict``, with keys being the tag name,
        and dict values being the value to set the tag to.

        The ``set_tags`` method
        should be called only in the ``__init__`` method of an object,
        during construction, or directly after construction via ``__init__``.

        Current tag values can be inspected by ``get_tags`` or ``get_tag``.

        Parameters
        ----------
        **tag_dict : dict
            Dictionary of tag name: tag value pairs.

        Returns
        -------
        Self
            Reference to self.
        """
        self._deprecate_tag_warn(tag_dict.keys())

        tag_dict = self._complete_dict(tag_dict, direction="old_to_new")
        self._set_flags(flag_attr_name="_tags", **tag_dict)
        return self

    @classmethod
    def _complete_dict(cls, tag_dict, direction="both"):
        """Add all aliased and aliasing tags to the dictionary.

        Parameters
        ----------
        tag_dict : dict
            Dictionary of tag name: tag value pairs.
        direction : str, one of "old_to_new", "both"
            Direction of aliasing to complete the dictionary for.

            * "old_to_new": complete only from old tags to new tags
            * "both": complete both from old to new and from new to old
        """
        alias_dict = cls.alias_dict
        deprecated_tags = set(tag_dict.keys()).intersection(alias_dict.keys())
        new_tags = set(tag_dict.keys()).intersection(alias_dict.values())

        if len(deprecated_tags) > 0 or len(new_tags) > 0:
            new_tag_dict = deepcopy(tag_dict)
            # for all tag strings being set, write the value
            #   to all tags that could *be aliased by* the string
            #   and all tags that could be *aliasing* the string
            # this way we ensure upwards and downwards compatibility
            for old_tag in alias_dict:
                cls._translate_tags(new_tag_dict, tag_dict, old_tag, direction)
            return new_tag_dict
        else:
            return tag_dict

    @classmethod
    def _translate_tags(cls, new_tag_dict, tag_dict, old_tag, direction="both"):
        """Translate old tag to new tag.

        Mutates ``new_tag_dict`` given ``old_tag_dict`` and ``old_tag``.

        Parameters
        ----------
        new_tag_dict : dict
            Dictionary of new tags.
        tag_dict : dict
            Dictionary of old tags.
        old_tag : str
            Name of the tag to translate.

        Returns
        -------
        str
            Translated tag name.
        """
        alias_dict = cls.alias_dict
        new_tag = alias_dict[old_tag]

        # todo 1.0.0 - remove this special case
        # special treatment for tags that get boolean flipped:
        # "ignores-exogeneous-X", "univariate-only"
        # the new tag is the negation of the old tag
        if old_tag in cls.FLIPPED_TAGS:
            if old_tag in tag_dict and new_tag != "":
                new_tag_dict[new_tag] = not tag_dict[old_tag]
            if direction == "both" and new_tag in tag_dict and old_tag not in tag_dict:
                new_tag_dict[old_tag] = not tag_dict[new_tag]
            if direction == "old_to_new" and old_tag in new_tag_dict:
                del new_tag_dict[old_tag]
            return new_tag_dict

        # standard treatment for all other tags
        if old_tag in tag_dict and new_tag != "":
            new_tag_dict[new_tag] = tag_dict[old_tag]
        if direction == "both" and new_tag in tag_dict and old_tag not in tag_dict:
            new_tag_dict[old_tag] = tag_dict[new_tag]
        if direction == "old_to_new" and old_tag in new_tag_dict:
            del new_tag_dict[old_tag]
        return new_tag_dict

    # package name used for deprecation warnings
    _package_name = "sktime"

    FLIPPED_TAGS = ["ignores-exogeneous-X", "univariate-only"]


# todo 1.0.0: remove TagAliaserMixin from inheritance
# remove redundant methods from sktime class (compare skbase)
class BaseEstimator(TagAliaserMixin, _BaseEstimator, BaseObject):
    """Base class for defining estimators in sktime.

    Extends sktime's BaseObject to include basic functionality for fittable estimators.
    """

    # tuple of non-BaseObject classes that count as nested objects
    # get_fitted_params will retrieve parameters from these, too
    # override in descendant class
    # _SklearnBaseEstimator = sklearn.base.BaseEstimator
    GET_FITTED_PARAMS_NESTING = (_SklearnBaseEstimator,)


def _clone_estimator(base_estimator, random_state=None):
    estimator = clone(base_estimator)

    if random_state is not None:
        set_random_state(estimator, random_state)

    return estimator


def _safe_clone(object):
    """Clone an object.

    If the object has a clone method, use that.

    Otherwise delegates to sklearn's clone function.
    """
    if hasattr(object, "clone"):
        return object.clone()
    else:
        return clone(object)


def deepcopy_func(f, name=None):
    """Deepcopy of a function."""
    import types

    return types.FunctionType(
        f.__code__,
        f.__globals__,
        name or f.__name__,
        f.__defaults__,
        f.__closure__,
    )


# initialize dynamic docstrings
BaseObject._init_dynamic_doc()
