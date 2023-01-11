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
            "list" - valid values are all lists of arbitrary elements
            ("str", list_of_string) - any string in list_of_string is valid
            ("list", list_of_string) - any individual string and sub-list is valid
            ("list", "str") - any individual string or list of strings is valid
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

__author__ = ["fkiraly", "victordremov"]

import pandas as pd

ESTIMATOR_TAG_REGISTER = [
    (
        "ignores-exogeneous-X",
        "forecaster",
        "bool",
        "does forecaster ignore exogeneous data (X)?",
    ),
    (
        "univariate-only",
        "transformer",
        "bool",
        "can transformer handle multivariate series? True = no",
    ),
    (
        "fit_is_empty",
        "estimator",
        "bool",
        "fit contains no logic and can be skipped? Yes=True, No=False",
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
        ["forecaster", "regressor"],
        "bool",
        "do X/y in fit/update and X/fh in predict have to be same indices?",
    ),
    (
        "enforce_index_type",
        ["forecaster", "regressor"],
        "type",
        "passed to input checks, input conversion index type to enforce",
    ),
    (
        "symmetric",
        ["transformer-pairwise", "transformer-pairwise-panel"],
        "bool",
        "is the transformer symmetric, i.e., t(x,y)=t(y,x) always?",
    ),
    (
        "scitype:X",
        "param_est",
        "str",
        "which scitypes does X internally support?",
    ),
    (
        "scitype:y",
        "forecaster",
        ("str", ["univariate", "multivariate", "both"]),
        "which series type does the forecaster support? multivariate means >1 vars",
    ),
    (
        "y_inner_mtype",
        ["forecaster", "transformer"],
        (
            "list",
            [
                "pd.Series",
                "pd.DataFrame",
                "np.array",
                "nested_univ",
                "pd-multiindex",
                "numpy3D",
                "df-list",
            ],
        ),
        "which machine type(s) is the internal _fit/_predict able to deal with?",
    ),
    (
        "X_inner_mtype",
        ["forecaster", "transformer", "transformer-pairwise-panel", "param_est"],
        (
            "list",
            [
                "pd.Series",
                "pd.DataFrame",
                "np.array",
                "nested_univ",
                "pd-multiindex",
                "numpy3D",
                "df-list",
            ],
        ),
        "which machine type(s) is the internal _fit/_predict able to deal with?",
    ),
    (
        "scitype:transform-input",
        "transformer",
        ("list", ["Series", "Panel"]),
        "what is the scitype of the transformer input X",
    ),
    (
        "scitype:transform-output",
        "transformer",
        ("list", ["Series", "Primitives", "Panel"]),
        "what is the scitype of the transformer output, the transformed X",
    ),
    (
        "scitype:instancewise",
        "transformer",
        "bool",
        "does the transformer transform instances independently?",
    ),
    (
        "scitype:transform-labels",
        "transformer",
        ("list", ["None", "Series", "Primitives", "Panel"]),
        "what is the scitype of y: None (not needed), Primitives, Series, Panel?",
    ),
    (
        "requires_y",
        "transformer",
        "bool",
        "does this transformer require y to be passed in fit and transform?",
    ),
    (
        "capability:inverse_transform",
        "transformer",
        "bool",
        "is the transformer capable of carrying out an inverse transform?",
    ),
    (
        "capability:pred_int",
        "forecaster",
        "bool",
        "does the forecaster implement predict_interval or predict_quantiles?",
    ),
    (
        "capability:pred_var",
        "forecaster",
        "bool",
        "does the forecaster implement predict_variance?",
    ),
    (
        "capability:multivariate",
        [
            "classifier",
            "early_classifier",
            "param_est",
            "regressor",
            "transformer-pairwise",
            "transformer-pairwise-panel",
        ],
        "bool",
        "can the classifier classify time series with 2 or more variables?",
    ),
    (
        "capability:unequal_length",
        [
            "classifier",
            "early_classifier",
            "regressor",
            "transformer",
            "transformer-pairwise-panel",
        ],
        "bool",
        "can the estimator handle unequal length time series?",
    ),
    # "capability:missing_values" is same as "handles-missing-data" tag.
    # They are kept distinct intentionally for easier TSC refactoring.
    # Will be merged after refactor completion.
    (
        "capability:missing_values",
        [
            "classifier",
            "early_classifier",
            "param_est",
            "regressor",
            "transformer-pairwise",
            "transformer-pairwise-panel",
        ],
        "bool",
        "can the classifier handle missing data (NA, np.nan) in inputs?",
    ),
    (
        "capability:unequal_length:removes",
        "transformer",
        "bool",
        "is the transformer result guaranteed to be equal length series (and series)?",
    ),
    (
        "capability:missing_values:removes",
        "transformer",
        "bool",
        "is the transformer result guaranteed to have no missing values?",
    ),
    (
        "capability:train_estimate",
        "classifier",
        "bool",
        "can the classifier estimate its performance on the training set?",
    ),
    (
        "capability:contractable",
        "classifier",
        "bool",
        "contract time setting, does the estimator support limiting max fit time?",
    ),
    (
        "capability:multithreading",
        ["classifier", "early_classifier"],
        "bool",
        "can the classifier set n_jobs to use multiple threads?",
    ),
    (
        "classifier_type",
        "classifier",
        (
            "list",
            [
                "dictionary",
                "distance",
                "feature",
                "hybrid",
                "interval",
                "kernel",
                "shapelet",
            ],
        ),
        "which type the classifier falls under in the taxonomy of time series "
        "classification algorithms.",
    ),
    (
        "capability:multiple-alignment",
        "aligner",
        "bool",
        "is aligner capable of aligning multiple series (True) or only two (False)?",
    ),
    (
        "capability:distance",
        "aligner",
        "bool",
        "does aligner return overall distance between aligned series?",
    ),
    (
        "capability:distance-matrix",
        "aligner",
        "bool",
        "does aligner return pairwise distance matrix between aligned series?",
    ),
    (
        "requires-y-train",
        "metric",
        "bool",
        "does metric require y-train data to be passed?",
    ),
    (
        "requires-y-pred-benchmark",
        "metric",
        "bool",
        "does metric require a predictive benchmark?",
    ),
    (
        "univariate-metric",
        "metric",
        "bool",
        "Does the metric only work on univariate y data?",
    ),
    (
        "scitype:y_pred",
        "metric",
        "str",
        "What is the scitype of y_pred: quantiles, proba, interval?",
    ),
    (
        "lower_is_better",
        "metric",
        "bool",
        "Is a lower value better for the metric? True=yes, False=higher is better",
    ),
    (
        "inner_implements_multilevel",
        "metric",
        "bool",
        "whether inner _evaluate can deal with multilevel (Panel/Hierarchical)",
    ),
    (
        "python_version",
        "estimator",
        "str",
        "python version specifier (PEP 440) for estimator, or None = all versions ok",
    ),
    (
        "python_dependencies",
        "estimator",
        ("list", "str"),
        "python dependencies of estimator as str or list of str",
    ),
    (
        "remember_data",
        ["forecaster", "transformer"],
        "bool",
        "whether estimator remembers all data seen as self._X, self._y, etc",
    ),
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

    if tag_type == "list" and not isinstance(tag_value, list):
        raise ValueError(tag_name + " must be list, found " + tag_value)

    if tag_type[0] == "str" and tag_value not in tag_type[1]:
        raise ValueError(
            tag_name + " must be one of " + tag_type[1] + " found " + tag_value
        )

    if tag_type[0] == "list" and not set(tag_value).issubset(tag_type[1]):
        raise ValueError(
            tag_name + " must be subest of " + tag_type[1] + " found " + tag_value
        )

    if tag_type[0] == "list" and tag_type[1] == "str":
        msg = f"{tag_name} must be str or list of str, found {tag_value}"
        if not isinstance(tag_value, (str, list)):
            raise ValueError(msg)
        if isinstance(tag_value, list):
            if not all(isinstance(x, str) for x in tag_value):
                raise ValueError(msg)
