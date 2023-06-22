# -*- coding: utf-8 -*-
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
    """
    pattern = r"\b([A-Z][A-Za-z0-9_]*)\b"
    cls_name_list = re.findall(pattern, spec)
    return cls_name_list


def craft(spec):
    """Get all maximal alphanumeric substrings that start with a capital letter.

    Parameters
    ----------
    spec : str, sktime/skbase compatible object specification
        i.e., a string that executes to construct an object if all imports were present
        imports inferred are of any classes in the scope of ``all_estimators``

    Returns
    -------
    obj : skbase BaseObject descendant, constructed from ``spec``
        this will have the property that ``spec == str(obj)`` (up to formatting)
    """
    register = dict(all_estimators())  # noqa: F841

    for x in _extract_class_names(spec):
        exec(f"{x} = register['{x}']")

    try:
        obj = eval(spec)
    except Exception:
        from textwrap import indent

        spec_fun = indent(spec, "    ")
        spec_fun = """
def build_obj():
        """ + spec_fun
        exec(spec_fun, locals())
        obj = eval("build_obj()")

    return obj
