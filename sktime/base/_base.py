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

    Extends skbase BaseObject with additional features.
    """

    # global default tags for dependency management
    _tags = {
        "python_version": None,  # PEP 440 version specifier, e.g., ">=3.7"
        "python_dependencies": None,  # PEP 440 dependency strs, e.g., "pandas>=1.0"
        "env_marker": None,  # PEP 508 environment marker, e.g., "os_name=='posix'"
        "sktime_version": SKTIME_VERSION,  # current sktime version
        # default tags for testing
        "tests:core": False,  # core objects have wider trigger conditions in testing
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
        if ``path`` is None, returns an in-memory serialized self
        if ``path`` is a file location, stores self at that location as a zip file

        saved files are zip files with following contents:
        _metadata - contains class of self, i.e., type(self)
        _obj - serialized self. This class uses the default serialization (pickle).

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
    """

    alias_dict = {"handles-missing-data": "capability:missing_values"}
    deprecate_dict = {"handles-missing-data": "1.0.0"}

    # package name used for deprecation warnings
    _package_name = "sktime"


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
