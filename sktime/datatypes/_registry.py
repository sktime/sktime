# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Registry of mtypes and scitypes.

Note for extenders: new mtypes for an existing scitypes
    should be entered in the _registry in the module with name _[scitype].

This module exports the following:

---

SCITYPE_REGISTER - list of tuples

each tuple corresponds to an scitype tag, elements as follows:
    0 : string - name of the scitype as used throughout sktime and in datatypes
    1 : string - plain English description of the scitype

---

MTYPE_REGISTER - list of tuples

each tuple corresponds to an mtype, elements as follows:
    0 : string - name of the mtype as used throughout sktime and in datatypes
    1 : string - name of the scitype the mtype is for, must be in SCITYPE_REGISTER
    2 : string - plain English description of the scitype

---

generate_mtype_soft_deps(scitype: str = None) - function returning dict

returns dictionary mapping mtypes with soft dependencies to values (str or list of str)
strings in values are names of soft dependency packages required for the mtype

---

mtype_to_scitype(mtype: str) - convenience function that returns scitype for an mtype

---
"""

from functools import lru_cache


def _only_core_deps(cls):
    """Return True if the class has only core dependencies."""
    # dependencies implied by the core requirements
    # that may appear in the classes as python_dependencies tag
    # it is assumed that only simple dependency strings, no version requirements appear
    DEPS_PRESENT_IN_ENV = ["numpy", "pandas"]

    deps_tag = cls.get_tag("python_dependencies")

    if deps_tag is None:
        return True
    if not isinstance(deps_tag, (list, tuple)) and deps_tag in DEPS_PRESENT_IN_ENV:
        return True
    if isinstance(deps_tag, (list, tuple)):
        return all([x in DEPS_PRESENT_IN_ENV for x in deps_tag])
    return False


def generate_mtype_cls_list(soft_deps="present"):
    """Generate list of mtype classes using lookup.

    Parameters
    ----------
    softdeps : str, optional, default = "present"
        how inclusion in relation to presence of soft dependencies is handled

        * "exclude" = only classes that do not require soft dependencies are returned
        * "present" = only classes with soft deps satisfied by the current python
        environment are returned
        * "all" = all classes, irrespective of soft deps satisfied or required, returned
        any other value defaults to "all"

    Returns
    -------
    classes : list of classes
        all classes that are mtypes, i.e. subclasses of BaseDatatype
        and not starting with "Base" or "Scitype"
    """
    if soft_deps not in ["exclude", "present", "all"]:
        raise ValueError(
            "Error in soft_deps argument in generate_mtype_cls_list: "
            "soft_deps must be one of 'exclude', 'present', 'all', "
            f"but got {soft_deps}"
        )
    return _generate_mtype_cls_list(soft_deps=soft_deps).copy()


@lru_cache(maxsize=3)
def _generate_mtype_cls_list(soft_deps="present"):
    """Generate list of mtype classes using lookup, cached function."""
    from skbase.utils.dependencies import _check_estimator_deps

    from sktime.datatypes._base import BaseDatatype
    from sktime.utils.retrieval import _all_classes

    classes = _all_classes("sktime.datatypes")
    classes = [x[1] for x in classes]
    classes = [x for x in classes if issubclass(x, BaseDatatype)]
    classes = [x for x in classes if not x.__name__.startswith("Base")]
    classes = [x for x in classes if not x.__name__.startswith("Scitype")]

    # subset only to data types with soft dependencies present
    if soft_deps == "present":
        classes = [x for x in classes if _check_estimator_deps(x, severity="none")]
    elif soft_deps == "exclude":
        classes = [x for x in classes if _only_core_deps(x)]
    # elif soft_deps=="all", no filtering happens

    return classes


def generate_mtype_register(scitype=None, soft_deps="all"):
    """Generate mtype class register using lookup.

    Parameters
    ----------
    scitype str or None, optional, default = None
        optional scitype to restrict the mtypes to

        * if None, all mtypes are returned
        * if str, must be scitype string, only mtypes for the scitype are returned

    softdeps : str, optional, default = "all"
        how inclusion in relation to presence of soft dependencies is handled

        * "exclude" = only classes that do not require soft dependencies are returned
        * "present" = only classes with soft deps satisfied by the current python
        environment are returned
        * "all" = all classes, irrespective of soft deps satisfied or required, returned
        any other value defaults to "all"

    Returns
    -------
    register : list of tuples
        each tuple corresponds to an mtype, elements as follows:

        0 : string - name of the mtype as used throughout sktime and in datatypes

        1 : string - name of the scitype the mtype is for, must be in SCITYPE_REGISTER

        2 : string - plain English description of the scitype
    """
    return _generate_mtype_register_subset(scitype=scitype, soft_deps=soft_deps)


@lru_cache(maxsize=256)
def _generate_mtype_register_subset(scitype=None, soft_deps="all"):
    """Generate mtype class register using lookup, cached function."""
    full_register = _generate_mtype_register(soft_deps=soft_deps)
    if scitype is None:
        return full_register.copy()
    else:
        return [x for x in full_register if x[1] == scitype]


@lru_cache(maxsize=3)
def _generate_mtype_register(soft_deps="all"):
    """Generate mtype class register using lookup, cached function."""
    classes = _generate_mtype_cls_list(soft_deps=soft_deps)

    def to_tuple(cls):
        """Return tuple of mtype register elements."""
        mtype = cls.get_class_tag("name")
        scitype = cls.get_class_tag("scitype")
        desc = cls.get_class_tag("description")
        return mtype, scitype, desc

    register = [to_tuple(x) for x in classes]
    return register


def generate_scitype_cls_list(soft_deps="present"):
    """Generate list of scitype classes using lookup.

    Parameters
    ----------
    soft_deps : str, optional, default="present"
        How inclusion in relation to presence of soft dependencies is handled.

        * "exclude" : only classes that do not require soft dependencies are returned.
        * "present" : only classes with soft deps satisfied by the current python
          environment are returned.
        * "all" : all classes, irrespective of soft deps satisfied or required,
          are returned. Any other value defaults to "all".

    Returns
    -------
    classes : list of classes
        All classes that represent scitypes, i.e., subclasses of BaseDatatype
        that start with "Scitype".
    """
    if soft_deps not in ["exclude", "present", "all"]:
        raise ValueError(
            "Error in soft_deps argument in generate_scitype_cls_list: "
            "soft_deps must be one of 'exclude', 'present', 'all', "
            f"but got {soft_deps}"
        )
    return _generate_scitype_cls_list(soft_deps=soft_deps).copy()


@lru_cache(maxsize=3)
def _generate_scitype_cls_list(soft_deps="present"):
    """Generate list of scitype classes using lookup, cached function."""
    from skbase.utils.dependencies import _check_estimator_deps

    from sktime.datatypes._base import BaseDatatype
    from sktime.utils.retrieval import _all_classes

    classes = _all_classes("sktime.datatypes")
    classes = [x[1] for x in classes]
    classes = [x for x in classes if issubclass(x, BaseDatatype)]
    classes = [x for x in classes if x.__name__.startswith("Scitype")]

    # subset only to data types with soft dependencies present
    if soft_deps == "present":
        classes = [x for x in classes if _check_estimator_deps(x, severity="none")]
    elif soft_deps == "exclude":
        classes = [x for x in classes if _only_core_deps(x)]
    # elif soft_deps=="all", no filtering happens

    return classes


def generate_scitype_register(scitype=None, soft_deps="all"):
    """Generate scitype register using dynamic lookup.

    Parameters
    ----------
    scitype : str or None, optional, default=None
        Optional scitype string to restrict the output to.

        * If None, all scitypes are returned.
        * If str, only the specified scitype is returned (if it exists).

    soft_deps : str, optional, default="all"
        How inclusion in relation to presence of soft dependencies is handled.

        * "exclude" : only scitypes originating from classes that do not require
          soft dependencies are returned.
        * "present" : only scitypes originating from classes with soft deps
          satisfied by the current python environment are returned.
        * "all" : all scitypes, irrespective of soft deps satisfied or required,
          are returned. Any other value defaults to "all".

    Returns
    -------
    register : list of tuples
        Each tuple corresponds to a recognized scitype, elements as follows:

        0 : str - name of the scitype as used throughout the library.
        1 : str - plain English description of the scitype.
    """
    return _generate_scitype_register_subset(scitype=scitype, soft_deps=soft_deps)


@lru_cache(maxsize=256)
def _generate_scitype_register_subset(scitype=None, soft_deps="all"):
    """Generate scitype class register using lookup, cached function."""
    full_register = _generate_scitype_register(soft_deps=soft_deps)
    if scitype is None:
        return full_register.copy()
    else:
        return [x for x in full_register if x[0] == scitype]


@lru_cache(maxsize=3)
def _generate_scitype_register(soft_deps="all"):
    """Generate scitype class register using lookup, cached function."""
    classes = _generate_scitype_cls_list(soft_deps=soft_deps)

    def to_tuple(cls):
        """Return tuple of scitype register elements."""
        scitype = cls.get_class_tag("scitype")
        desc = cls.get_class_tag("description")
        return scitype, desc

    register = [to_tuple(x) for x in classes]
    return register


def generate_mtype_list(scitype=None, soft_deps="all"):
    """Generate mtype list using lookup.

    Parameters
    ----------
    scitype str or None, optional, default = None
        optional scitype to restrict the mtypes to

        * if None, all mtypes are returned
        * if str, must be scitype string, only mtypes for the scitype are returned

    softdeps : str, optional, default = "all"
        how inclusion in relation to presence of soft dependencies is handled

        * "exclude" = only classes that do not require soft dependencies are returned
        * "present" = only classes with soft deps satisfied by the current python
        environment are returned
        * "all" = all classes, irrespective of soft deps satisfied or required, returned
        any other value defaults to "all"

    Returns
    -------
    register : list of string
        entries are name of the mtype as used throughout sktime and in datatypes
    """
    return [x[0] for x in generate_mtype_register(scitype=scitype, soft_deps=soft_deps)]


def generate_mtype_soft_deps(scitype=None):
    """Generate mtype python dependencies using lookup.

    Parameters
    ----------
    scitype : str or None, optional, default = None
        optional scitype to restrict the mtypes to

        * if None, dependencies for all mtypes are returned
        * if str, must be a valid scitype string; only dependencies for mtypes
          belonging to that scitype are returned

    Returns
    -------
    soft_deps : dict
        dictionary mapping mtypes to their soft dependencies
        keys : str - name of the mtype as used throughout sktime
        values : str or list of str - names of soft dependency packages required
    """
    return _generate_mtype_soft_deps_subset(scitype=scitype)


@lru_cache(maxsize=256)
def _generate_mtype_soft_deps_subset(scitype=None):
    """Generate mtype soft dependencies filtered by scitype, cached function."""
    full_soft_deps = _generate_mtype_soft_deps()
    filtered_list = []
    if scitype is None:
        filtered_list = full_soft_deps.copy()
    else:
        filtered_list = [x for x in full_soft_deps if x[1] == scitype]

    deps_dict = {}

    for mtype, _, soft_deps in filtered_list:
        deps_dict[mtype] = soft_deps

    return deps_dict


@lru_cache(maxsize=2)
def _generate_mtype_soft_deps():
    """Generate mtype soft dependencies, cached function."""
    classes = _generate_mtype_cls_list(soft_deps="all")
    deps_list = []

    for cls in classes:
        soft_deps = cls.get_class_tag("python_dependencies", None)

        if soft_deps is not None:
            mtype = cls.get_class_tag("name")
            scitype = cls.get_class_tag("scitype")
            deps_list.append((mtype, scitype, soft_deps))

    return deps_list


# mtypes to exclude in checking since they are ambiguous and rare
AMBIGUOUS_MTYPES = ["numpyflat", "alignment_loc", "pd-long", "pd-wide"]


def mtype_to_scitype(mtype: str, return_unique=False, coerce_to_list=False):
    """Infer scitype belonging to mtype.

    Parameters
    ----------
    mtype : str, or list of str, or nested list/str object, or None
        mtype(s) to find scitype of, a valid mtype string
        valid mtype strings, with explanation, are in datatypes.MTYPE_REGISTER

    Returns
    -------
    scitype : str, or list of str, or nested list/str object, or None
        if str, returns scitype belonging to mtype, if mtype is str
        if list, returns this function element-wise applied
        if nested list/str object, replaces mtype str by scitype str
        if None, returns None
    return_unique : bool, default=False
        if True, makes return unique
    coerce_to_list : bool, default=False
        if True, coerces rerturn to list, even if one-element

    Raises
    ------
    TypeError, if input is not of the type specified
    ValueError, if there are two scitypes for the/some mtype string
        (this should not happen in general, it means there is a bug)
    ValueError, if there is no scitype for the/some mtype string
    """
    # handle the "None" case first
    if mtype is None or mtype == "None":
        return None
    # recurse if mtype is a list
    if isinstance(mtype, list):
        scitype_list = [mtype_to_scitype(x) for x in mtype]
        if return_unique:
            scitype_list = list(set(scitype_list))
        return scitype_list

    # checking for type. Checking str is enough, recursion above will do the rest.
    if not isinstance(mtype, str):
        raise TypeError(
            "mtype must be str, or list of str, nested list/str object, or None"
        )

    scitype = [k[1] for k in generate_mtype_register() if k[0] == mtype]

    if len(scitype) > 1:
        raise ValueError("multiple scitypes match the mtype, specify scitype")

    if len(scitype) < 1:
        raise ValueError(f"{mtype} is not a supported mtype")

    if coerce_to_list:
        return [scitype[0]]
    else:
        return scitype[0]


def scitype_to_mtype(scitype: str, softdeps: str = "exclude"):
    """Return list of all mtypes belonging to scitype.

    Parameters
    ----------
    scitype : str, or list of str
        scitype(s) to find mtypes for, a valid scitype string
        valid scitype strings, with explanation, are in datatypes.SCITYPE_REGISTER
    softdeps : str, optional, default = "exclude"
        whether to return mtypes that require soft dependencies
        "exclude" = only mtypes that do not require soft dependencies are returned
        "present" = only mtypes with soft deps satisfied by the environment are returned
        "all" = all mtypes, irrespective of soft deps satisfied or required, returned
        any other value defaults to "all"

    Returns
    -------
    mtypes : list of str
        all mtypes such that mtype_to_scitype(element) is in the list scitype
        if list, returns this function element-wise applied
        if nested list/str object, replaces mtype str by scitype str
        if None, returns None

    Raises
    ------
    TypeError, if input is not of the type specified
    ValueError, if one of the strings is not a valid scitype string
    RuntimeError, if there is no mtype for the/some scitype string (this must be a bug)
    """
    msg = "scitype argument must be str or list of str"
    # handle the "None" case first
    if scitype is None or scitype == "None":
        raise TypeError(msg)
    # recurse if mtype is a list
    if isinstance(scitype, list):
        scitype_list = [y for x in scitype for y in scitype_to_mtype(x)]
        return scitype_list

    # checking for type. Checking str is enough, recursion above will do the rest.
    if not isinstance(scitype, str):
        raise TypeError(msg)

    # now we know scitype is a string, check if it is in the register
    SCITYPE_LIST = [x[0] for x in generate_scitype_register()]
    if scitype not in SCITYPE_LIST:
        raise ValueError(
            f'"{scitype}" is not a valid scitype string, see datatypes.SCITYPE_REGISTER'
        )

    mtypes = [k[0] for k in generate_mtype_register() if k[1] == scitype]

    if len(mtypes) == 0:
        # if there are no mtypes, this must have been reached by mistake/bug
        raise RuntimeError("no mtypes defined for scitype " + scitype)

    if softdeps not in ["exclude", "present"]:
        return mtypes

    MTYPE_SOFT_DEPS = generate_mtype_soft_deps()
    if softdeps == "exclude":
        # subset to mtypes that require no soft deps
        mtypes = [m for m in mtypes if m not in MTYPE_SOFT_DEPS.keys()]
        return mtypes

    if softdeps == "present":
        from skbase.utils.dependencies import _check_soft_dependencies

        def present(x):
            """Return True if x has satisfied soft dependency or has no soft dep."""
            if x not in MTYPE_SOFT_DEPS.keys():
                return True
            else:
                return _check_soft_dependencies(MTYPE_SOFT_DEPS[x], severity="none")

        # return only mtypes with soft dependencies present (or requiring none)
        mtypes = [m for m in mtypes if present(m)]
        return mtypes
