# -*- coding: utf-8 -*-
"""Register of estimator and object tags.

Note for extenders: new tags should be entered in ESTIMATOR_TAG_REGISTER.
No other place is necessary to add new tags.

This module exports the following:

---

ESTIMATOR_TAG_REGISTER - list of tuples

each tuple corresponds to a tag, elements as follows:
    0 : string - name of the tag as used in the _tags dictionary
    1 : string - name of the scitype this tag applies to
                 must be in _base_classes.BASE_CLASS_SCITYPE_LIST
    2 : string - expected type of the tag value
        should be one of:
            "bool" - valid values are True/False
            "int" - valid values are all integers
            "str" - valid values are all strings
            ("str", list_of_string) - any string in list_of_string is valid
            ("list", list_of_string) - any individual string and sub-list is valid
        validity can be checked by check_tag_is_valid (see below)
    3 : string - plain English description of the tag

---

ESTIMATOR_TAG_TABLE - pd.DataFrame
    ESTIMATOR_TAG_REGISTER in table form, as pd.DataFrame
        rows of ESTIMATOR_TABLE correspond to elements in ESTIMATOR_TAG_REGISTER

ESTIMATOR_TAG_LIST - list of string
    elements are 0-th entries of ESTIMATOR_TAG_REGISTER, in same order

---

check_tag_is_valid(tag_name, tag_value) - checks whether tag_value is valid for tag_name

"""

__author__ = ["fkiraly"]

import pandas as pd


ESTIMATOR_TAG_REGISTER = [
    (
        "univariate-only",  # todo: rename to "scitype:handles_exogeneous"
        "forecaster",
        "bool",
        "does forecaster use exogeneous data (X)?",
    ),
    (
        "fit-in-transform",
        "transformer",
        "bool",
        "does fit contain no logic and can be skipped? yes/no",
    ),
    (
        "transform-returns-same-time-index",
        "transformer",
        "bool",
        "does transform return same time index as input?",
    ),
    (
        "handles-missing-data",
        "estimator",
        "bool",
        "can the estimator handle missing data (NA, np.nan) in inputs?",
    ),
    (
        "skip-inverse-transform",
        "transformer",
        "bool",
        "behaviour flag: skips inverse_transform when called yes/no",
    ),
    (
        "requires-fh-in-fit",
        "forecaster",
        "bool",
        "does forecaster require fh passed already in fit? yes/no",
    ),
    (
        "X-y-must-have-same-index",
        ["forecaster", "classifier", "regressor"],
        "bool",
        "do X/y in fit/update and X/fh in predict have to be same indices?",
    ),
    (
        "enforce-index-type",
        ["forecaster", "classifier", "regressor"],
        "type",
        "passed to input checks, input conversion index type to enforce",
    ),
    (
        "coerce-X-to-numpy",
        ["forecaster", "classifier", "regressor"],
        "bool",
        "should X be coerced to numpy type in check_X? yes/no",
    ),
    (
        "symmetric",
        ["transformer-pairwise-tabular", "transformer-pairwise-panel"],
        "bool",
        "is the transformer symmetric, i.e., t(x,y)=t(y,x) always?",
    ),
    (
        "scitype:y",
        "forecaster",
        ("str", ["univariate", "multivariate", "both"]),
        "which series type does the forecaster support? multivariate means >1 vars",
    ),
    (
        "y_inner_mtype",
        "forecaster",
        ("list", ["pd.Series", "pd.DataFrame", "np.array"]),
        "which machine type(s) is the internal _fit/_predict able to deal with?",
    ),
    (
        "X_inner_mtype",
        "forecaster",
        ("list", ["pd.Series", "pd.DataFrame", "np.array"]),
        "which machine type(s) is the internal _fit/_predict able to deal with?",
    ),
    (
        "capability:pred_int",
        "forecaster",
        "bool",
        "is the forecaster capable of returning prediction intervals in predict?",
    ),
    # (
    #     "handles-panel",
    #     "annotator",
    #     "bool",
    #     "can handle panel annotations, i.e., list X/y?",
    # ),
    # (
    #     "annotation-type",
    #     "annotator",
    #     "str",
    #     "which annotation type? can be 'point', 'segment' or 'both'",
    # ),
    # (
    #     "annotation-kind",
    #     "annotator",
    #     "str",
    #     "which annotations? can be 'outlier', 'change', 'label', 'none'",
    # ),
]

ESTIMATOR_TAG_TABLE = pd.DataFrame(ESTIMATOR_TAG_REGISTER)

ESTIMATOR_TAG_LIST = ESTIMATOR_TAG_TABLE[0].tolist()


def check_tag_is_valid(tag_name, tag_value):
    """Check validity of a tag value.

    Parameters
    ----------
    tag_name : string, name of the tag
    tag_value : object, value of the tag

    Raises
    ------
    KeyError - if tag_name is not a valid tag in ESTIMATOR_TAG_LIST
    ValueError - if the tag_valid is not a valid for the tag with name tag_name
    """
    if tag_name not in ESTIMATOR_TAG_LIST:
        raise KeyError(tag_name + " is not a valid tag")

    tag_type = ESTIMATOR_TAG_TABLE[2][ESTIMATOR_TAG_TABLE[0] == "tag_name"]

    if tag_type == "bool" and not isinstance(tag_value, bool):
        raise ValueError(tag_name + " must be True/False, found " + tag_value)

    if tag_type == "int" and not isinstance(tag_value, int):
        raise ValueError(tag_name + " must be integer, found " + tag_value)

    if tag_type == "str" and not isinstance(tag_value, str):
        raise ValueError(tag_name + " must be string, found " + tag_value)

    if tag_type[0] == "str" and tag_value not in tag_type[1]:
        raise ValueError(
            tag_name + " must be one of " + tag_type[1] + " found " + tag_value
        )

    if tag_type[0] == "list" and not set(tag_value).issubset(tag_type[1]):
        raise ValueError(
            tag_name + " must be subest of " + tag_type[1] + " found " + tag_value
        )
