# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Quick crafting methods to build an object from string and registry.

craft(spec)
    craft an object or estimator from string

deps(spec)
    retrieves all dependencies required to craft str, in PEP440 format

The ``craft`` function is a pair to ``str`` coercion, the two can be seen as
deserialization/serialization counterparts to each other.

That is,
spec = str(my_est)
new_est = craft(spec)

will have the same effect as new_est = spec.clone()
"""

__author__ = ["fkiraly"]

import re

from sktime.registry._lookup import all_estimators


def _extract_class_names(spec):
    """Get all maximal alphanumeric substrings that start with a capital letter.

    Parameters
    ----------
    spec : str (any)

    Returns
    -------
    cls_name_list : list of str
        list of all maximal alphanumeric substrings starting with a capital in ``spec``
        excluding reserved expressions True, False (if they occur)
    """
    # class names are all UpperCamelCase alphanumeric strings
    # which is the same as maximal substrings starting with a capital
    pattern = r"\b([A-Z][A-Za-z0-9_]*)\b"
    cls_name_list = re.findall(pattern, spec)

    # we need to exclude expressions that look like classes per the regex
    # but aren't
    EXCLUDE_LIST = ["True", "False"]
    cls_name_list = [x for x in cls_name_list if x not in EXCLUDE_LIST]

    return cls_name_list


def craft(spec):
    """Instantiate an object from the specification string.

    Parameters
    ----------
    spec : str, sktime/skbase compatible object specification
        i.e., a string that executes to construct an object if all imports were present
        imports inferred are of any classes in the scope of ``all_estimators``
        option 1: a string that evaluates to an estimator
        option 2: a sequence of assignments in valid python code,
            with the object to be defined preceded by a "return"
            assignments can use names of classes as if all imports were present

    Returns
    -------
    obj : skbase BaseObject descendant, constructed from ``spec``
        this will have the property that ``spec == str(obj)`` (up to formatting)
    """
    register = dict(all_estimators())  # noqa: F841

    try:
        obj = eval(spec, globals(), register)
    except Exception:
        from textwrap import indent

        spec_fun = indent(spec, "    ")
        spec_fun = (
            """
def build_obj():
        """
            + spec_fun
        )

        exec(spec_fun, register, register)
        obj = eval("build_obj()", register, register)

    return obj


def deps(spec):
    """Get PEP 440 dependency requirements for a craft spec.

    Parameters
    ----------
    spec : str, sktime/skbase compatible object specification
        i.e., a string that executes to construct an object if all imports were present
        imports inferred are of any classes in the scope of ``all_estimators``
        option 1: a string that evaluates to an estimator
        option 2: a sequence of assignments in valid python code,
            with the object to be defined preceded by a "return"
            assignments can use names of classes as if all imports were present

    Returns
    -------
    reqs : list of str
        each str is PEP 440 compatible requirement string for craft(spec)
        if spec has no requirements, return is [], the length 0 list
    """
    register = dict(all_estimators())

    dep_strs = []

    for x in _extract_class_names(spec):
        if x not in register.keys():
            raise RuntimeError(
                f"class {x} is required to build spec, but was not found "
                "in all_estimators scope"
            )
        cls = register[x]

        new_deps = cls.get_class_tag("python_dependencies")

        if isinstance(new_deps, list):
            dep_strs += new_deps
        elif isinstance(new_deps, str) and len(new_deps) > 0:
            dep_strs += [new_deps]

        reqs = list(set(dep_strs))

    return reqs


def imports(spec):
    """Get import code block for a craft spec.

    Parameters
    ----------
    spec : str, sktime/skbase compatible object specification
        i.e., a string that executes to construct an object if all imports were present
        imports inferred are of any classes in the scope of ``all_estimators``
        option 1: a string that evaluates to an estimator
        option 2: a sequence of assignments in valid python code,
            with the object to be defined preceded by a "return"
            assignments can use names of classes as if all imports were present

    Returns
    -------
    import_str : str
        python code consisting of all import statements required for spec
        imports cover object/estimator classes found as sub-strings of spec
    """
    register = dict(all_estimators())

    import_strs = []

    for x in _extract_class_names(spec):
        if x not in register.keys():
            raise RuntimeError(
                f"class {x} is required to build spec, but was not found "
                "in all_estimators scope"
            )
        cls = register[x]

        import_module = _get_public_import(cls.__module__)
        import_str = f"from {import_module} import {x}"
        import_strs += [import_str]

    if len(import_strs) == 0:
        imports_str = ""
    else:
        imports_str = "\n".join(sorted(import_strs))

    return imports_str


def _get_public_import(module_path: str) -> str:
    """Get the public import path from full import path.

    Removes everything from the first private submodule (starting with '_') onwards.
    """
    parts = module_path.split(".")
    for i, part in enumerate(parts):
        if part.startswith("_"):
            return ".".join(parts[:i])  # Keep only part before first private submodule
    return module_path  # Return the original path if no private submodules are found
