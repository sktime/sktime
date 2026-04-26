"""Register of estimator and object tags.

Note for extenders: new tags should be entered in OBJECT_TAG_REGISTER.
No other place is necessary to add new tags.

This module exports the following:

---
OBJECT_TAG_REGISTER - list of tuples

each tuple corresponds to a tag, elements as follows:
    0 : string - name of the tag as used in the _tags dictionary
    1 : string - name of the scitype this tag applies to
                 must be in _base_classes.BASE_CLASS_SCITYPE_LIST
    2 : string - expected type of the tag value
        should be one of:
            "bool" - valid values are True/False
            "int" - valid values are all integers
            "str" - valid values are all strings
            "list" - valid values are all lists of arbitrary elements
            ("str", list_of_string) - any string in list_of_string is valid
            ("list", list_of_string) - any individual string and sub-list is valid
            ("list", "str") - any individual string or list of strings is valid
        validity can be checked by check_tag_is_valid (see below)
    3 : string - plain English description of the tag

---

OBJECT_TAG_TABLE - pd.DataFrame
    OBJECT_TAG_REGISTER in table form, as pd.DataFrame
        rows of OBJECT_TABLE correspond to elements in OBJECT_TAG_REGISTER

OBJECT_TAG_LIST - list of string
    elements are 0-th entries of OBJECT_TAG_REGISTER, in same order

---

check_tag_is_valid(tag_name, tag_value) - checks whether tag_value is valid for tag_name
"""

OBJECT_TAG_REGISTER = [
    # --------------------------
    # all objects and estimators
    # --------------------------
    (
        "object_type",
        "object",
        "str",
        "type of object, e.g., 'regressor', 'transformer'",
    ),
    (
        "python_version",
        "object",
        "str",
        "python version specifier (PEP 440) for estimator, or None = all versions ok",
    ),
    (
        "python_dependencies",
        "object",
        ("list", "str"),
        "python dependencies of estimator as str or list of str",
    ),
    (
        "python_dependencies_alias",
        "object",
        "dict",
        "should be provided if import name differs from package name, \
        key-value pairs are package name, import name",
    ),
    # -----------------------
    # BaseTimeSeriesBootstrap
    # -----------------------
    (
        "bootstrap_type",
        "bootstrap",
        ("list", "str"),
        "which type of bootstrap the algorithm is",
    ),
    (
        "capability:multivariate",
        "bootstrap",
        "bool",
        "whether the bootstrap algorithm supports multivariate data",
    ),
    # ----------------------------
    # BaseMetaObject reserved tags
    # ----------------------------
    (
        "named_object_parameters",
        "object",
        "str",
        "name of component list attribute for meta-objects",
    ),
    (
        "fitted_named_object_parameters",
        "estimator",
        "str",
        "name of fitted component list attribute for meta-objects",
    ),
]

OBJECT_TAG_LIST = [x[0] for x in OBJECT_TAG_REGISTER]
