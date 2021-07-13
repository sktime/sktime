# -*- coding: utf-8 -*-
"""
Register of estimator and object tags.

This module exports the following:

---

ESTIMATOR_TAG_REGISTER - list of tuples

each tuple corresponds to a tag, elements as follows:
    0 : string - name of the tag as used in the _tags dictionary
    1 : string - name of the scitype this tag applies to
    2 : string - expected type of the tag value
    3 : string - plain English description of the tag

---

ESTIMATOR_TAG_LIST - list of string
    elements are 0-th entries of ESTIMATOR_TAG_REGISTER, in same order

"""

__author__ = ["fkiraly"]

import pandas as pd


ESTIMATOR_TAG_REGISTER = [
    (
        "univariate-only",
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

ESTIMATOR_TAG_LIST = pd.DataFrame(ESTIMATOR_TAG_REGISTER)[0].tolist()
