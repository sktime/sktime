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

import warnings
from copy import deepcopy

from skbase.base import BaseObject as _BaseObject
from sklearn import clone
from sklearn.base import BaseEstimator as _BaseEstimator

from sktime.exceptions import NotFittedError
from sktime.utils.random_state import set_random_state

SERIALIZATION_FORMATS = {
    "pickle",
    "cloudpickle",
}


class BaseObject(_BaseObject):
    """Base class for parametric objects with tags in sktime.

    Extends skbase BaseObject with additional features.
    """

    # global default tags for dependency management
    _tags = {
        "python_version": None,  # PEP 440 version specifier, e.g., ">=3.7"
        "python_dependencies": None,  # PEP 440 dependency strs, e.g., "pandas>=1.0"
        "python_dependencies_alias": {"scikit-learn": "sklearn"},
        "env_marker": None,  # PEP 508 environment marker, e.g., "os_name=='posix'"
    }

    _config = {
        "warnings": "on",
        "backend:parallel": None,  # parallelization backend for broadcasting
        #  {None, "dask", "loky", "multiprocessing", "threading"}
        #  None: no parallelization
        #  "loky", "multiprocessing" and "threading": uses `joblib` Parallel loops
        #  "dask": uses `dask`, requires `dask` package in environment
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

            - "None": executes loop sequentally, simple list comprehension
            - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel``
            - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
            - "dask": uses ``dask``, requires ``dask`` package in environment
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
        """,
    }

    # TODO 0.32.0: check whether python 3.8 has reached EoL.
    # If so, remove warning altogether
    def __init__(self):
        super().__init__()

        import sys

        from packaging.specifiers import SpecifierSet

        from sktime.utils.warnings import warn

        py39_or_higher = SpecifierSet(">=3.9")
        sys_version = sys.version.split(" ")[0]

        # todo 0.32.0 - check whether python 3.8 eol is reached.
        # If yes, remove this msg.
        if sys_version not in py39_or_higher:
            warn(
                f"From sktime 0.30.0, sktime requires Python version >=3.9, "
                f"but found {sys_version}. "
                "The package can still be installed, until 3.8 end of life "
                "is reached, "
                "but some functionality may not work as test coverage is dropped."
                "Kindly note for context: python 3.8 will reach end of life "
                "in October 2024, and multiple sktime core dependencies, "
                "including scikit-learn, have already dropped support for 3.8. ",
                category=DeprecationWarning,
                obj=self,
                stacklevel=2,
            )

        # handle numpy 2 incompatible soft dependencies
        # for rationale, see _handle_numpy2_softdeps
        self._handle_numpy2_softdeps()

    # TODO 0.32.0: check list of numpy 2 incompatible soft deps
    # remove any from NOT_NP2_COMPATIBLE that become compatible
    def _handle_numpy2_softdeps(self):
        """Handle tags for soft deps that are not numpy 2 compatible.

        A number of soft dependencies are not numpy 2 compatible yet,
        but do not set the bound in their setup.py. This method is a patch over
        those packages' missing bound setting to provide informative
        errors to users.

        This method does the following:

        * checks if any soft dependencies in the python_dependencies tag
          are in NOT_NP2_COMPATIBLE, this is a hard-coded
          list of soft dependencies that are not numpy 2 compatible
        * if any are found, adds a numpy<2.0 soft dependency to the list,
          and sets it as a dynamic override of the python_dependencies tag
        """
        from packaging.requirements import Requirement

        # pypi package names of soft dependencies that are not numpy 2 compatibleS
        NOT_NP2_COMPATIBLE = ["prophet", "numba"]

        softdeps = self.get_class_tag("python_dependencies", [])
        if softdeps is None:
            return None
        if not isinstance(softdeps, list):
            softdeps = [softdeps]
        # make copy of list to avoid side effects
        softdeps = softdeps.copy()

        def _pkg_name(req):
            """Get package name from requirement string."""
            return Requirement(req).name

        noncomp = False
        for softdep in softdeps:
            # variable: does any softdep string start with one of the non-compatibles
            noncomp_sd = any([_pkg_name(softdep) == pkg for pkg in NOT_NP2_COMPATIBLE])
            noncomp = noncomp or noncomp_sd

        if noncomp:
            softdeps = softdeps + ["numpy<2.0"]
            self.set_tags(python_dependencies=softdeps)
        return None

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
                path="estimator" then a zip file ``estimator.zip`` will be made at cwd.
                path="/home/stored/estimator" then a zip file ``estimator.zip`` will be
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


class TagAliaserMixin:
    """Mixin class for tag aliasing and deprecation of old tags.

    To deprecate tags, add the TagAliaserMixin to BaseObject or BaseEstimator.
    alias_dict contains the deprecated tags, and supports removal and renaming.     For
    removal, add an entry "old_tag_name": ""     For renaming, add an entry
    "old_tag_name": "new_tag_name" deprecate_dict contains the version number of
    renaming or removal.     the keys in deprecate_dict should be the same as in
    alias_dict.     values in deprecate_dict should be strings, the version of
    removal/renaming.

    The class will ensure that new tags alias old tags and vice versa, during the
    deprecation period. Informative warnings will be raised whenever the deprecated tags
    are being accessed.

    When removing tags, ensure to remove the removed tags from this class. If no tags
    are deprecated anymore (e.g., all deprecated tags are removed/renamed), ensure
    toremove this class as a parent of BaseObject or BaseEstimator.
    """

    def __init__(self):
        super().__init__()

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
        collected_tags = super().get_class_tags()
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
            Value of the ``tag_name`` tag in self. If not found, returns
            ``tag_value_default``.
        """
        cls._deprecate_tag_warn([tag_name])
        return super().get_class_tag(
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
        collected_tags = super().get_tags()
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
            Value of the ``tag_name`` tag in self. If not found, returns an error if
            raise_error is True, otherwise it returns ``tag_value_default``.

        Raises
        ------
        ValueError if raise_error is True i.e. if tag_name is not in self.get_tags(
        ).keys()
        """
        self._deprecate_tag_warn([tag_name])
        return super().get_tag(
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
        Changes object state by setting tag values in tag_dict as dynamic tags
        in self.
        """
        self._deprecate_tag_warn(tag_dict.keys())

        tag_dict = self._complete_dict(tag_dict)
        super().set_tags(**tag_dict)
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
                msg = f"tag {tag_name!r} will be removed in sktime version {version}"
                if new_tag != "":
                    msg += (
                        f" and replaced by {new_tag!r}, please use {new_tag!r} instead"
                    )
                else:
                    msg += ', please remove code that access or sets "{tag_name}"'
                warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


class BaseEstimator(BaseObject):
    """Base class for defining estimators in sktime.

    Extends sktime's BaseObject to include basic functionality for fittable estimators.
    """

    def __init__(self):
        self._is_fitted = False
        super().__init__()

    @property
    def is_fitted(self):
        """Whether ``fit`` has been called."""
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

    def get_fitted_params(self, deep=True):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        deep : bool, default=True
            Whether to return fitted parameters of components.

            * If True, will return a dict of parameter name : value for this object,
              including fitted parameters of fittable components
              (= BaseEstimator-valued parameters).
            * If False, will return a dict of parameter name : value for this object,
              but not include fitted parameters of components.

        Returns
        -------
        fitted_params : dict with str-valued keys
            Dictionary of fitted parameters, paramname : paramvalue
            keys-value pairs include:

            * always: all fitted parameters of this object, as via ``get_param_names``
              values are fitted parameter value for that key, of this object
            * if ``deep=True``, also contains keys/value pairs of component parameters
              parameters of components are indexed as ``[componentname]__[paramname]``
              all parameters of ``componentname`` appear as ``paramname`` with its value
            * if ``deep=True``, also contains arbitrary levels of component recursion,
              e.g., ``[componentname]__[componentcomponentname]__[paramname]``, etc
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"estimator of type {type(self).__name__} has not been "
                "fitted yet, please call fit on data before get_fitted_params"
            )

        # collect non-nested fitted params of self
        fitted_params = self._get_fitted_params()

        # the rest is only for nested parameters
        # so, if deep=False, we simply return here
        if not deep:
            return fitted_params

        def sh(x):
            """Shorthand to remove all underscores at end of a string."""
            if x.endswith("_"):
                return sh(x[:-1])
            else:
                return x

        # add all nested parameters from components that are sktime BaseObject
        c_dict = self._components()
        for c, comp in c_dict.items():
            if isinstance(comp, BaseEstimator) and comp._is_fitted:
                c_f_params = comp.get_fitted_params()
                c_f_params = {f"{sh(c)}__{k}": v for k, v in c_f_params.items()}
                fitted_params.update(c_f_params)

        # add all nested parameters from components that are sklearn estimators
        # we do this recursively as we have to reach into nested sklearn estimators
        n_new_params = 42
        old_new_params = fitted_params
        while n_new_params > 0:
            new_params = dict()
            for c, comp in old_new_params.items():
                if isinstance(comp, _BaseEstimator):
                    c_f_params = self._get_fitted_params_default(comp)
                    c_f_params = {f"{sh(c)}__{k}": v for k, v in c_f_params.items()}
                    new_params.update(c_f_params)
            fitted_params.update(new_params)
            old_new_params = new_params.copy()
            n_new_params = len(new_params)

        return fitted_params

    def _get_fitted_params_default(self, obj=None):
        """Obtain fitted params of object, per sklearn convention.

        Extracts a dict with {paramstr : paramvalue} contents,
        where paramstr are all string names of "fitted parameters".

        A "fitted attribute" of obj is one that ends in "_" but does not start with "_".
        "fitted parameters" are names of fitted attributes, minus the "_" at the end.

        Parameters
        ----------
        obj : any object, optional, default=self

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        obj = obj if obj else self

        # default retrieves all self attributes ending in "_"
        # and returns them with keys that have the "_" removed
        fitted_params = {
            attr[:-1]: getattr(obj, attr)
            for attr in dir(obj)
            if attr.endswith("_") and not attr.startswith("_") and hasattr(obj, attr)
        }
        return fitted_params

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
        return self._get_fitted_params_default()


def _clone_estimator(base_estimator, random_state=None):
    estimator = clone(base_estimator)

    if random_state is not None:
        set_random_state(estimator, random_state)

    return estimator


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
