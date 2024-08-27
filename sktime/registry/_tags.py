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

import inspect
import sys

import pandas as pd

from sktime.base import BaseObject
from sktime.registry._base_classes import BASE_CLASS_REGISTER


class _BaseTag(BaseObject):
    """Base class for all tags."""

    _tags = {
        "object_type": "tag",
        "tag_name": "fill_this_in",  # name of the tag used in the _tags dictionary
        "parent_type": "object",  # scitype of the parent object, str or list of str
        "tag_type": "str",  # type of the tag value
        "short_descr": "describe the tag here",  # short tag description, max 80 chars
        "user_facing": True,  # whether the tag is user-facing
    }


# General tags, for all objects
# -----------------------------


class object_type(_BaseTag):
    """Scientific type of the object.

    Typing tag for all objects in ``sktime``.

    - String name: ``"object_type"``
    - Public metadata tag
    - Values:  string or list of strings
    - Example: ``"forecaster"``
    - Example 2: ``["transformer", "clusterer"]`` (polymorphic object)
    - Default: ``"object"``

    In ``sktime``, every object has a scientific type (scitype),
    determining the type of object and unified interface,
    e.g., forecaster, time series classifier, time series regressor.

    The ``object_type`` tag of an object is a string, or list of strings,
    specifying the scitpye of the object.
    For instance, a forecaster has scitype `"forecaster"`.

    In case of a list, the object is polymorphic, and can assume (class),
    or simultaneously satisfy different interfaces (object).

    Valid scitypes are defined in ``sktime.registry.BASE_CLASS_SCITYPE_LIST``,
    or ``sktime.registry.BASE_CLASS_REGISTER``.

    The full list of scitypes in the current version is:
    """

    _tags = {
        "tag_name": "object_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "type of object: estimator, transformer, regressor, etc",
        "user_facing": True,
    }


# dynamically add a pretty printd list of scitypes to the docstring
for name, _, desc in BASE_CLASS_REGISTER:
    object_type.__doc__ += f'\n    - ``"{name}"``: {desc}'


class maintainers(_BaseTag):
    """Current maintainers of the object, GitHub IDs.

    Part of packaging metadata for the object.

    - String name: ``"maintainers"``
    - Public metadata tag
    - Values:  string or list of strings
    - Example: ``["benheid", "fkiraly", "yarnabrina"]``
    - Example 2: ``"yarnabrina"``
    - Default: ``"sktime developers"``

    The ``maintainers`` tag of an object is a string or list of strings,
    each string being a GitHub handle of a maintainer of the object.

    Maintenance extends to the specific class in ``sktime`` only,
    and not interfaced packages or dependencies.

    Maintainers should be tagged on issues and PRs related to the object,
    and have rights and responsibilities in accordance with
    the ``sktime`` governance model, see :ref:`algorithm-maintainers`.

    To find an algorithm's maintainer, use ``get_tag("maintainers")`` on the object,
    or use the search function in the
    `estimator overview <https://www.sktime.net/en/stable/estimator_overview.html>`_.

    In case of classes not owned by specific algorithm maintainers,
    the tag defaults to the ``sktime`` core team.

    Maintainers are given prominent visibility in the object's metadata,
    and in automatically generated documentation.
    """

    _tags = {
        "tag_name": "maintainers",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "current maintainers of the object, each maintainer a GitHub handle",  # noqa: E501
        "user_facing": True,
    }


class authors(_BaseTag):
    """Authors of the object, GitHub IDs.

    Part of packaging metadata for the object.

    - String name: ``"authors"``
    - Public metadata tag
    - Values:  string or list of strings
    - Example: ``["benheid", "fkiraly", "yarnabrina"]``
    - Example 2: ``"fkiraly"``
    - Default: ``"sktime developers"``

    The ``authors`` tag of an object is a string or list of strings,
    each string being a GitHub handle of an author of the object.

    Authors are credited for the original implementation of the object,
    and contributions to the object.

    In case of light wrappers around third or second party packages,
    author credits should include authors of the wrapped object.

    Authors are not necessarily maintainers of the object,
    and do not need to be tagged on issues and PRs related to the object.

    Authors are given prominent visibility in the object's metadata,
    and in automatically generated documentation.

    To find an algorithm's authors, use ``get_tag("authors")`` on the object,
    or use the search function in the
    `estimator overview <https://www.sktime.net/en/stable/estimator_overview.html>`_.
    """

    _tags = {
        "tag_name": "authors",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of authors of the object, each author a GitHub handle",
        "user_facing": True,
    }


class python_version(_BaseTag):
    """Python version requirement specifier for the object (PEP 440).

    Part of packaging metadata for the object.

    - String name: ``"python_version"``
    - Private tag, developer and framework facing
    - Values: PEP 440 compliant version specifier
    - Example: ``">=3.10"``
    - Default: no restriction

    ``sktime`` manages objects and estimators like mini-packages,
    with their own dependencies and compatibility requirements.
    Dependencies are specified in the tags:

    - ``"python_version"``: Python version specifier (PEP 440) for the object,
    - ``"python_dependencies"``: list of required Python packages (PEP 440)
    - ``"env_marker"``: environment marker for the object (PEP 508)
    - ``"requires_cython"``: whether the object requires a C compiler present

    The ``python_version`` tag of an object is a PEP 440 compliant version specifier
    string, specifying python version compatibility of the object.

    The tag is used in packaging metadata for the object,
    and is used internally to check compatibility of the object with
    the build environment, to raise informative error messages.

    Developers can use ``_check_python_version`` from ``skbase.utils.dependencies``
    to check compatibility of the python constraint of the object
    with the current build environment, or
    ``_check_estimator_deps`` to check compatibility of the object
    (including further checks) with the current build environment.
    """

    _tags = {
        "tag_name": "python_version",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "python version specifier (PEP 440) for estimator, or None = all versions ok",  # noqa: E501
        "user_facing": False,
    }


class python_dependencies(_BaseTag):
    """Python package dependency requirement specifiers for the object (PEP 440).

    Part of packaging metadata for the object.

    - String name: ``"python_dependencies"``
    - Private tag, developer and framework facing
    - Values: str or list of str, each str a PEP 440 compliant dependency specifier
    - Example: ``"numpy>=1.20.0"``
    - Example 2: ``["numpy>=1.20.0", "pandas>=1.3.0"]``
    - Default: no requirements beyond ``sktime`` core dependencies (``None``)

    ``sktime`` manages objects and estimators like mini-packages,
    with their own dependencies and compatibility requirements.
    Dependencies are specified in the tags:

    - ``"python_version"``: Python version specifier (PEP 440) for the object,
    - ``"python_dependencies"``: list of required Python packages (PEP 440)
    - ``"env_marker"``: environment marker for the object (PEP 508)
    - ``"requires_cython"``: whether the object requires a C compiler present

    The ``python_dependencies`` tag of an object is string or list of strings,
    each string a PEP 440 compliant version specifier,
    specifying python dependency requirements of the object.

    The tag is used in packaging metadata for the object,
    and is used internally to check compatibility of the object with
    the build environment, to raise informative error messages.

    Developers should note that package names in PEP 440 specifier strings
    are identical with the package names used in ``pip install`` commands,
    which in general is not the same as the import name of the package,
    e.g., ``"scikit-learn"`` and not ``"sklearn"``.

    Developers can use ``_check_soft_dependencies`` from ``skbase.utils.dependencies``
    to check compatibility of the python constraint of the object
    with the current build environment, or
    ``_check_estimator_deps`` to check compatibility of the object
    (including further checks) with the current build environment.
    """

    _tags = {
        "tag_name": "python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "python dependencies of estimator as str or list of str (PEP 440)",  # noqa: E501
        "user_facing": False,
    }


class env_marker(_BaseTag):
    """Environment marker requirement for the object (PEP 508).

    Part of packaging metadata for the object.

    - String name: ``"env_marker"``
    - Private tag, developer and framework facing
    - Values: str, PEP 508 compliant environment marker
    - Example: ``"platform_system == 'Linux'"``
    - Default: no environment marker (``None``)

    ``sktime`` manages objects and estimators like mini-packages,
    with their own dependencies and compatibility requirements.
    Dependencies are specified in the tags:

    - ``"python_version"``: Python version specifier (PEP 440) for the object,
    - ``"python_dependencies"``: list of required Python packages (PEP 440)
    - ``"env_marker"``: environment marker for the object (PEP 508)
    - ``"requires_cython"``: whether the object requires a C compiler present

    The ``env_marker`` tag of an object is a string,
    specifying a PEP 508 compliant environment marker for the object.

        The tag is used in packaging metadata for the object,
    and is used internally to check compatibility of the object with
    the build environment, to raise informative error messages.

    Developers can use ``_check_env_marker`` from ``skbase.utils.dependencies``
    to check compatibility of the python constraint of the object
    with the current build environment, or
    ``_check_estimator_deps`` to check compatibility of the object
    (including further checks) with the current build environment.
    """

    _tags = {
        "tag_name": "env_marker",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "environment marker (PEP 508) requirement for estimator, or None = no marker",  # noqa: E501
        "user_facing": False,
    }


class requires_cython(_BaseTag):
    """Whether the object requires a C compiler present, such as libomp, gcc.

    Part of packaging metadata for the object.

    - String name: ``"requires_cython"``
    - Private tag, developer and framework facing
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    ``sktime`` manages objects and estimators like mini-packages,
    with their own dependencies and compatibility requirements.
    Dependencies are specified in the tags:

    - ``"python_version"``: Python version specifier (PEP 440) for the object,
    - ``"python_dependencies"``: list of required Python packages (PEP 440)
    - ``"env_marker"``: environment marker for the object (PEP 508)
    - ``"requires_cython"``: whether the object requires a C compiler present

    The ``requires_cython`` tag of an object is a boolean,
    specifying whether the object requires a C compiler present.
    True means that a C compiler is required, False means it is not required.

    The tag is used in packaging metadata for the object,
    and primarily in the continuous integration and testing setup of the ``sktime``
    package, which ensures that objects with this tag are
    tested in specific environments with a C compiler present.

    It is not used in user facing checks, error messages,
    or recommended build processes otherwise.
    """

    _tags = {
        "tag_name": "requires_cython",
        "parent_type": "object",
        "tag_type": "bool",
        "short_descr": "whether the object requires a C compiler present such as libomp, gcc",  # noqa: E501
        "user_facing": False,
    }


# Estimator tags
# --------------

# These tags are applicable to a wide range of objects,
# most tags in this group apply to estimators

# "capability:missing_values" is same as "handles-missing-data" tag.
# They are kept distinct intentionally for easier TSC refactoring.
# Will be merged after refactor completion.


class capability__missing_values(_BaseTag):
    """Capability: the estimator can handle missing data, e.g,, NaNs.

    - String name: ``"capability:missing_values"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``
    - Alias: ``handles-missing-data``  (forecasters, transformations)

    If the tag is ``True``, the estimator can handle missing data,
    e.g., NaNs in input data.

    This applies to main and secondary input data where applicable,
    e.g., ``X`` in ``fit`` of transformations and classifieres, or ``y`` in forecasters,
    but not to target labels
    in the case of labelling of entire time series, such as in
    classification or regression.

    If the tag is ``False``, the estimator cannot handle missing data,
    and will raise an error if missing data is encountered.
    """

    _tags = {
        "tag_name": "capability:missing_values",
        "parent_type": "object",
        "tag_type": "bool",
        "short_descr": "can the estimator handle missing data (NA, np.nan) in inputs?",  # noqa: E501
        "user_facing": True,
    }


class capability__feature_importance(_BaseTag):
    """Capability: the estimator can provide feature importance.

    - String name: ``"capability:feature_importance"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    If the tag is ``True``, the estimator can produce feature importances.

    Feature importances are queryable by the fitted parameter interface
    via ``get_fitted_params``, after calling ``fit`` of the respective estimator.

    If the tag is ``False``, the estimator does not produce feature importances.
    The method ``get_fitted_params`` can be called,
    but the list of fitted parameters will not contain feature importances.
    """

    _tags = {
        "tag_name": "capability:feature_importance",
        "parent_type": "estimator",
        "tag_type": "bool",
        "short_descr": "Can the estimator provide feature importance?",
        "user_facing": True,
    }


class capability__contractable(_BaseTag):
    """Capability: the estimator can be asked to satisfy a maximum time contract.

    To avoid confusion, users should note that the literature term "contractable"
    unusually derives its meaning from "contract", i.e., a time contract,
    and not from "contraction", i.e., reducing the size of something.

    - String name: ``"capability:contractable"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    If the tag is ``True``, the estimator can be contracted in time,
    by using some of its parameters, to limit the maximum time spent in fitting.

    Currently, there is no unified naming or selection of parameters controlling
    the contract time setting, as this can apply to different parts of the algorithm.
    Users should consult the documentation of the specific estimator for details.

    If the tag is ``False``, the estimator does not support a contract time setting.
    """

    _tags = {
        "tag_name": "capability:contractable",
        "parent_type": "estimator",
        "tag_type": "bool",
        "short_descr": "contract time setting, does the estimator support limiting max fit time?",  # noqa: E501
        "user_facing": True,
    }


class capability__train_estimate(_BaseTag):
    """Capability: the algorithm can estimate its performance on the training set.

    - String name: ``"capability:train_estimate"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    If the tag is ``True``, the estimator can estimate its performance on
    the training set.

    More precisely, this tag describes algorithms that, when calling ``fit``,
    produce and store an estimate of their own statistical performance,
    e.g., via out-of-bag estimates, or cross-validation.

    Training performance estimates are queryable by the fitted parameter interface
    via ``get_fitted_params``, after calling ``fit`` of the respective estimator.

    If the tag is ``False``, the estimator does not produce
    training performance estimates.
    The method ``get_fitted_params`` can be called,
    but the list of fitted parameters will not contain training performance estimates.
    """

    _tags = {
        "tag_name": "capability:train_estimate",
        "parent_type": "estimator",
        "tag_type": "bool",
        "short_descr": "can the estimator estimate its performance on the training set?",  # noqa: E501
        "user_facing": True,
    }


class fit_is_empty(_BaseTag):
    """Property: Whether the estimator has an empty fit method.

    - String name: ``"fit_is_empty"``
    - Public property tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``True`` (transformations), ``False`` (other estimators)

    If the tag is ``True``, the estimator has an empty ``fit`` method,
    i.e., the method does not perform any calculations or learning.
    If the tag is ``False``, the estimator has a non-empty ``fit`` method.

    In both cases, calling ``fit`` is necessary for calling further methods
    such as ``predict`` or ``transform``, for API consistency.

    The tag may be inspected by the user to distinguish between estimators
    that do not learn from data from those that do.

    The tag is also used internally by ``sktime`` to short cut boilerplate
    code, e.g., in the ``fit`` methods.
    """

    _tags = {
        "tag_name": "fit_is_empty",
        "parent_type": "estimator",
        "tag_type": "bool",
        "short_descr": "does the estimator have an empty fit method?",
        "user_facing": True,
    }


# Forecasters
# -----------


class capability__exogeneous(_BaseTag):
    """Capability: the forecaster can use exogeneous data.

    The tag is currently named ``ignores-exogeneous-X``, and will be renamed.

    ``False`` = does use exogeneous data, ``True`` = does not use exogeneous data.

    - String name: ``"ignores-exogeneous-X"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``
    - Alias: ``capability:exogeneous`` (currently not used)

    Exogeneous data are additional time series,
    that can be used to improve forecasting accuracy.

    If the forecaster uses exogeneous data (``ignore-exogeneous-X=False``),
    the ``X`` parameter in ``fit``, ``predict``, and other methods
    can be used to pass exogeneous data to the forecaster.

    If the ``X-y-must-have-same-index`` tag is ``True``,
    then such data must always have an index that contains that of the target series,
    i.e., ``y`` in ``fit``, or the indices specified by ``fh`` in ``predict``.

    If the tag is ``False``, the forecaster does not make use of exogeneous data.
    ``X`` parameters can still be passed to methods, to ensure a uniform interface,
    but the data will be ignored,
    i.e., not used in the internal logic of the forecaster.

    """

    _tags = {
        "tag_name": "ignores-exogeneous-X",
        "parent_type": "forecaster",
        "tag_type": "bool",
        "short_descr": "does forecaster make use of exogeneous data?",
        "user_facing": True,
    }


class capability__insample(_BaseTag):
    """Capability: the forecaster can make in-sample predictions.

    - String name: ``"capability:insample"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    If the tag is ``True``, the forecaster can make in-sample predictions,
    i.e., predict the target series for time points that are part of the training set.

    In-sample predictions are useful for model evaluation,
    and for making predictions for the training set itself.

    Mechanically, in-sample predictions are made by calling the ``predict`` method
    and specifying a forecasting horizon ``fh`` such that at least one index
    is queried that is equal or earlier to the latest index in the training set,
    i.e., any data previously passed in ``fit`` or ``update``.

    If the tag is ``False``, the forecaster cannot make in-sample predictions,
    and will raise an error if an in-sample prediction is attempted.
    """

    _tags = {
        "tag_name": "capability:insample",
        "parent_type": "forecaster",
        "tag_type": "bool",
        "short_descr": "can the forecaster make in-sample predictions?",
        "user_facing": True,
    }


class capability__pred_int(_BaseTag):
    """Capability: the forecaster can make probabilistic or interval forecasts.

    - String name: ``"capability:pred_int"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    ``sktime`` supports a range of ways to make probabilistic type forecasts,
    via the following methods of any forecaster:

    * ``predict_interval``: prediction intervals
    * ``predict_quantiles``: quantile forecasts
    * ``predict_var``: variance forecasts
    * ``predict_proba``: distribution forecasts

    If the ``capability:pred_int`` tag is ``True``, the forecaster can make
    probabilistic type forecasts using all of the above methods.

    Even if the forecaster natively implements only one of the above methods,
    all are available to the user:

    * interval and quantile forecasts are of equivalent information,
      with intervals of a coverage assumed symmetric
    * a forecaster with available distribution forecasts obtains
      prediction intervals and quantile forecasts from the distribution
    * a forecaster with available variance forecasts assumes a normal distribution
      around the ``predict`` output as mean,
      and derives prediction intervals and quantiles
      from that normal distribution
    * a forecaster with available interval or quantile uses the IQR to
      to derive variance forecasts under normality assumptions.
      Users should note that this may lead to a distribution forecast which
      is not consistent with interval or quantile forecasts.

    If the tag is ``False``, the forecaster cannot make probabilistic forecasts,
    and will raise an error if a probabilistic forecast is attempted.
    """

    _tags = {
        "tag_name": "capability:pred_int",
        "parent_type": "forecaster",
        "tag_type": "bool",
        "short_descr": "does the forecaster implement predict_interval or predict_quantiles?",  # noqa: E501
        "user_facing": True,
    }


class capability__pred_int__insample(_BaseTag):
    """Capability: the forecaster can make in-sample probabilistic forecasts.

    Only relevant if the ``capability:pred_int`` tag is ``True``,
    i.e., the forecaster can make probabilistic forecasts.

    - String name: ``"capability:pred_int:insample"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``False``
    - Default: ``True``

    If the tag is ``True``, the forecaster can make
    its probabilistic forecasts in-sample, i.e.,
    ``predict_interval``, ``predict_quantiles``, ``predict_var``,
    or ``predict_proba`` can be called with a forecasting horizon ``fh``
    that includes in-sample indices, i.e., indices that are not later than
    the latest index in the training set.

    If the tag ``capability:pred_int`` is ``False``,
    then the tag ``capability:pred_int:insample`` is irrelevant,
    as the forecaster cannot make probabilistic forecasts at all.
    In such a case, the tag ``capability:pred_int:insample`` should be ignored.

    If the tag ``capability:pred_int`` is ``True``,
    and is the tag ``capability:pred_int:insample`` is ``False``,
    the forecaster can make probabilistic forecasts that are out-of-sample,
    but cannot make in-sample probabilistic forecasts,
    and will raise an error if an in-sample probabilistic forecast is attempted.
    """

    _tags = {
        "tag_name": "capability:pred_int:insample",
        "parent_type": "forecaster",
        "tag_type": "bool",
        "short_descr": "can the forecaster make in-sample predictions in predict_interval/quantiles?",  # noqa: E501
        "user_facing": True,
    }


class requires_fh_in_fit(_BaseTag):
    """Behaviour flag: forecaster requires forecasting horizon in fit.

    - String name: ``"requires-fh-in-fit"``
    - Public behaviour flag
    - Values: boolean, ``True`` / ``False``
    - Example: ``False``
    - Default: ``True``

    If the tag is ``True``, the forecaster requires the forecasting horizon
    to be passed in the ``fit`` method, i.e., the ``fh`` argument must be non-``None``.

    If the tag is ``False``, the forecasting horizon can be passed in the ``fit``
    method, but this is not required. In this case, it must be passed later,
    whenever ``predict`` or other prediction methods are called.

    Whether the ``fh`` is required in ``fit`` is an intrinsic property of the
    forecasting algorithm and not a user setting.

    For instance, direct reduction to tabular regression
    requires the ``fh`` as it is used by the fitting algorithm to lag the endogeneous
    against the exogeneous data. In contrast, recursive reduction to tabular regression
    does not require the ``fh`` in ``fit``, as only the prediction step
    requires the forecasting horizon, when applying the fitted tabular regression model
    by sliding it forward over the ``fh`` steps.
    """

    _tags = {
        "tag_name": "requires-fh-in-fit",
        "parent_type": "forecaster",
        "tag_type": "bool",
        "short_descr": "does the forecaster require the forecasting horizon in fit?",  # noqa: E501
        "user_facing": True,
    }


class capability__categorical_in_X(_BaseTag):
    """Capability: If estimator can handle categorical natively in exogeneous(X) data.

    ``False`` = cannot handle categorical natively in X,
    ``True`` = can handle categorical natively in X

    - String name: ``"capability:categorical_in_X"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    Exogeneous data are additional time series,
    that can be used to improve forecasting accuracy.
    """

    _tags = {
        "tag_name": "capability:categorical_in_X",
        "parent_type": ["forecaster", "transformer"],
        "tag_type": "bool",
        "short_descr": "can the estimator natively handle categorical data in exogeneous X?",  # noqa: E501
        "user_facing": True,
    }


# Panel data related tags
# -----------------------

# tags related to panel data, typically:
# classification, regression, clustering, and transformations


class capability__multivariate(_BaseTag):
    """Capability: the estimator can handle multivariate time series.

    - String name: ``"capability:multivariate"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``
    - Alias: ``univariate-only``  (transformations, note: boolean is inverted)
    - Alias: ``univariate-metric`` (performance metrics, note: boolean is inverted)

    If the tag is ``True``, the estimator can handle multivariate time series,
    for its main input data, i.e., the ``X`` parameter in ``fit`` of classifiers,
    regressors, clusterers, ordinary transformers, and pairwise transformers.

    If the tag is ``False``, the estimator can only handle univariate time series,
    and will broadcast to variables (ordinary transformers), or raise an error (others).

    This condition is specific to the main input data representation,
    target data (e.g., classifier or transformation ``y``) are not considered.

    The condition is also specific to the data type used, in terms of how
    being "multivariate" is represented.
    For instance, a ``pandas`` based time series specification is considered
    multivariate if it has more than one column.
    """

    _tags = {
        "tag_name": "capability:multivariate",
        "parent_type": [
            "classifier",
            "clusterer",
            "early_classifier",
            "param_est",
            "regressor",
            "transformer-pairwise",
            "transformer-pairwise-panel",
        ],
        "tag_type": "bool",
        "short_descr": "can the estimator be applied to time series with 2 or more variables?",  # noqa: E501
        "user_facing": True,
    }


class capability__unequal_length(_BaseTag):
    """Capability: the estimator can handle unequal length time series.

    - String name: ``"capability:unequal_length"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    Tag applicable to estimators which can accept panel data,
    i.e., collections of time series.

    If the tag is ``True``, the estimator can handle panels of time series
    with unequal index set, including panels of time series with
    unequal length.

    It should be noted that the capability implied by the tag is
    strictly more general than the capability implied by the name of the tag,
    as panels of time series of equal length can have unequal index sets.

    If the tag is ``False``, the estimator requires all time series in the panel
    to have equal length and index set, and will otherwise raise an error.
    """

    _tags = {
        "tag_name": "capability:unequal_length",
        "parent_type": [
            "aligner",
            "classifier",
            "clusterer",
            "early_classifier",
            "regressor",
            "transformer",
            "transformer-pairwise-panel",
        ],
        "tag_type": "bool",
        "short_descr": "can the estimator handle unequal length time series?",
        "user_facing": True,
    }


class capability__multioutput(_BaseTag):
    """Capability: the estimator can handle multi-output time series.

    - String name: ``"capability:multioutput"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    This tag applies to classifiers and regressors.

    If the tag is ``True``, the estimator can handle multivariate target time series,
    i.e., time series with multiple variables in the target argument ``y``.

    If the tag is ``False``, the estimator can only handle univariate targets natively,
    and will broadcast to variables (ordinary transformers), or raise an error (others).

    This condition is specific to target data (e.g., classifier or regressor ``y``),
    primary input data (``X``) is not considered.

    The capability for primary input data is controlled by the tag
    ``capability:multivariate``.

    The condition is also specific to the data type used, in terms of how
    being "multivariate" is represented.
    For instance, a ``pandas`` based time series specification is considered
    multivariate if it has more than one column.
    """

    _tags = {
        "tag_name": "capability:multioutput",
        "parent_type": ["classifier", "regressor"],
        "tag_type": "bool",
        "short_descr": "can the estimator handle multi-output time series?",
        "user_facing": True,
    }


class capability__predict(_BaseTag):
    """Capability: the clusterer can predict cluster assignments.

    - String name: ``"capability:predict"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``False``
    - Default: ``True``

    This tag applies to clusterers only.

    If the tag is ``True``, the clusterer implements a ``predict``
    method, which can be used to obtain cluster assignments.

    If the tag is ``False``, the clusterer will raise an exception on
    ``predict`` call.
    """

    _tags = {
        "tag_name": "capability:predict",
        "parent_type": "clusterer",
        "tag_type": "bool",
        "short_descr": (
            "can the clusterer predict cluster assignments for new data points?"
        ),
        "user_facing": True,
    }


class capability__predict_proba(_BaseTag):
    """Capability: the estimator can make probabilistic predictions.

    - String name: ``"capability:predict_proba"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    This tag applies to classifiers and clusterers.

    If the tag is ``True``, the estimator implements a non-default ``predict_proba``
    method, which can be used to predict class probabilities (classifier),
    or probabilistic cluster assignments (clusterer)

    If the tag is ``False``, the estimator's ``predict_proba`` defaults to
    predicting zero/one probabilities, equivalent to the ``predict`` output.
    """

    _tags = {
        "tag_name": "capability:predict_proba",
        "parent_type": ["classifier", "clusterer"],
        "tag_type": "bool",
        "short_descr": (
            "does the estimator implement a non-default predict_proba method? "
            "i.e., not just 0/1 probabilities obtained from predict?"
        ),
        "user_facing": True,
    }


class capability__out_of_sample(_BaseTag):
    """Capability: the estimator can make out-of-sample predictions.

    - String name: ``"capability:out_of_sample"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``False``
    - Default: ``True``

    This tag applies to clusterers only.

    If the tag is ``True``, the estimator can make cluster assignments
    out-of-sample, i.e., the indices in ``predict`` need not be equal to those
    seen in ``fit``.

    If the tag is ``False``, the estimator will refit a clone
    when ``predict`` is called, on the pooled data seen in ``fit`` and ``predict``,
    if there is ad least one index value in ``predict`` that has not been seen in
    ``fit``. For index-less data mtypes, identity of the data object is used to check
    whether indices are equal.
    """

    _tags = {
        "tag_name": "capability:out_of_sample",
        "parent_type": "clusterer",
        "tag_type": "bool",
        "short_descr": (
            "can the clusterer make out-of-sample predictions, "
            "i.e., compute cluster assignments on new data?"
        ),
        "user_facing": True,
    }


# Transformations
# ---------------


class scitype__transform_input(_BaseTag):
    """The scitype of the input data for the transformer.

    - String name: ``"scitype:transform-input"``
    - Public scitype tag
    - Values: string, one of ``"Series"`` or ``"Panel"``
    - Example: ``"Series"``
    - Default: ``"Series"``

    Transformations in ``sktime`` are polymorphic and can have one of multiple
    input/output behaviours, depending on the scitype of the input data.

    The following tags specify input/output behaviour:

    - ``"scitype:transform-input"``: the scitype of the input data ``X``.
    - ``"scitype:transform-output"``: the scitype of the output data, given the input.
    - ``"scitype:instancewise"``: whether the transformation is instance-wise.
    - ``"scitype:transform-labels"``: the scitype of the target labels ``y``, if used
    - ``"requires_X"``: whether ``X`` is mandatory in ``fit`` and ``transform``
    - ``"requires_y"``: whether ``y`` is mandatory in ``fit`` and ``transform``

    The tags ``"scitype:transform-input"`` and ``"scitype:transform-output"``
    together specify the input/output typing of the transformation.

    The possible values for both are from a list of :term:`scitype` strings, which are:

    * ``"Series"``: a single time series.
    * ``"Panel"``: a panel of time series, i.e., a collection of time series.
    * ``"Primitives"``: a collection of primitive types, e.g., a collection of scalars.
      This is an alias for the scitype ``"Table"`` used in the `datatypes` module.

    The combination of the two tags is to be read as:

    * if ``"scitype:transform-input"`` has value ``input_type``,
      and ``"scitype:transform-output"`` has value ``output_type``,
    * then, if I pass input data of scitype ``input_type`` to ``transform``
      of the transformer, I will get output data of scitype ``output_type``.
    * further input types are handled by broadcasting over instances or indices,
      where possible.

    For instance, if a transformer has ``"scitype:transform-input"`` being ``"Series"``,
    and ``"scitype:transform-output"`` being ``"Series"``, then ``transform`` will
    produce a single time series as output, given a single time series as input.

    Other input types are handled by broadcasting over instances or indices.
    For instance, in the same case, if the input a panel of time series
    (of scitype ``"Panel"``), then the transformer will transform each time series
    and produce an output panel of time series (of scipy ``"Panel"``).

    It should be noted that this is in the case where both tags have value
    ``"Series"``, the behaviour for ``"Panel"`` is implied by broadcasting.

    The value ``"Panel"`` is used only if the transformation adds index levels,
    or removes index levels, or changes the number or indices of series in the panel.

    Writing shorthand "Series-to-Series" for the type pair ``"Series"`` to ``"Series"``,
    and similarly for other types, the possible type pairs are listed below.

    For illustration, it is recommended to try out the transformations mentioned,
    on the respective input types, to understand the behaviour.

    * Series-to-Series, this transforms individual series to individual series.
      Panels are transformed to Panel, and Hierarchical series to Hierarchical series.
      Examples are lagging, ``Lag``, or differencing, ``Differencer``.
    * Series-to-Primitives, this transforms individual series to a collection of
      primitives, a single time series is transformed to a single row of a
      ``pd.DataFrame``. A panel is transformed to a ``pd.DataFrame``, with
      as many rows as time series in the panel.
      A hierarchical series is transformed to a ``pd.DataFrame`` with the hierarchy
      indices retained, one row corresponding to a non-temporal leaf node.
      Examples are feature extraction or summarization (mean, quantiles, etc), see
      ``SummaryTransformer``.
    * Series-to-Panel, this transforms individual series to a panel of time series.
      Panels are transformed to hierarchical series with added index levels.
      Examples are time series bootstrapping, where multiple bootstrap samples are
      produced per input series, see ``STLBootstrapTransformer``,
      or ``TSBootstrapAdapter``.
    * Panel-to-Series, this transforms a panel of time series to a single time series.
      Examples are aggregation with time index retained, e.g., mean per index or bin,
      see ``Merger``.

    The relationship between input and output types of ``transform`` is
    summarized in the following table,
    for the case where ``"scitype:transform-input"`` is ``"Series"``.

    The first column is the type of ``X``, which need not be ``"Series"``,
    the second column is the value of the ``"scitype:transform-output"`` tag,
    the third column is the type of the output of ``transform``.

    The output type is obtained from the input type of ``transform``, from
    broadcasting of the types defined by the tag values.

    .. list-table::
        :widths: 35 35 40
        :header-rows: 2

        * -
          - `transform`
          -
        * - `X`
          - `-output`
          - type of return
        * - `Series`
          - `Primitives`
          - `pd.DataFrame` (1-row)
        * - `Panel`
          - `Primitives`
          - `pd.DataFrame`
        * - `Series`
          - `Series`
          - `Series`
        * - `Panel`
          - `Series`
          - `Panel`
        * - `Series`
          - `Panel`
          - `Panel`

    The instance indices in the in return correspond to instances in the input ``X``.
    """

    _tags = {
        "tag_name": "scitype:transform-input",
        "parent_type": "transformer",
        "tag_type": ("str", ["Series", "Panel"]),
        "short_descr": "what is the scitype of the transformer input X?",
        "user_facing": True,
    }


class scitype__transform_output(_BaseTag):
    """The scitype of the input data for the transformer.

    - String name: ``"scitype:transform-output"``
    - Public scitype tag
    - Values: string, one of ``"Series"``, ``"Panel"``, ``"Primitives"``
    - Example: ``"Series"``
    - Default: ``"Series"``

    Transformations in ``sktime`` are polymorphic and can have one of multiple
    input/output behaviours, depending on the scitype of the input data.

    The following tags specify input/output behaviour:

    - ``"scitype:transform-input"``: the scitype of the input data ``X``.
    - ``"scitype:transform-output"``: the scitype of the output data, given the input.
    - ``"scitype:instancewise"``: whether the transformation is instance-wise.
    - ``"scitype:transform-labels"``: the scitype of the target labels ``y``, if used
    - ``"requires_X"``: whether ``X`` is mandatory in ``fit`` and ``transform``
    - ``"requires_y"``: whether ``y`` is mandatory in ``fit`` and ``transform``

    The tags ``"scitype:transform-input"`` and ``"scitype:transform-output"``
    together specify the input/output typing of the transformation.

    The possible values for both are from a list of :term:`scitype` strings, which are:

    * ``"Series"``: a single time series.
    * ``"Panel"``: a panel of time series, i.e., a collection of time series.
    * ``"Primitives"``: a collection of primitive types, e.g., a collection of scalars.
      This is an alias for the scitype ``"Table"`` used in the `datatypes` module.

    The combination of the two tags is to be read as:

    * if ``"scitype:transform-input"`` has value ``input_type``,
      and ``"scitype:transform-output"`` has value ``output_type``,
    * then, if I pass input data of scitype ``input_type`` to ``transform``
      of the transformer, I will get output data of scitype ``output_type``.
    * further input types are handled by broadcasting over instances or indices,
      where possible.

    For instance, if a transformer has ``"scitype:transform-input"`` being ``"Series"``,
    and ``"scitype:transform-output"`` being ``"Series"``, then ``transform`` will
    produce a single time series as output, given a single time series as input.

    Other input types are handled by broadcasting over instances or indices.
    For instance, in the same case, if the input a panel of time series
    (of scitype ``"Panel"``), then the transformer will transform each time series
    and produce an output panel of time series (of scipy ``"Panel"``).

    It should be noted that this is in the case where both tags have value
    ``"Series"``, the behaviour for ``"Panel"`` is implied by broadcasting.

    The value ``"Panel"`` is used only if the transformation adds index levels,
    or removes index levels, or changes the number or indices of series in the panel.

    Writing shorthand "Series-to-Series" for the type pair ``"Series"`` to ``"Series"``,
    and similarly for other types, the possible type pairs are listed below.

    For illustration, it is recommended to try out the transformations mentioned,
    on the respective input types, to understand the behaviour.

    * Series-to-Series, this transforms individual series to individual series.
      Panels are transformed to Panel, and Hierarchical series to Hierarchical series.
      Examples are lagging, ``Lag``, or differencing, ``Differencer``.
    * Series-to-Primitives, this transforms individual series to a collection of
      primitives, a single time series is transformed to a single row of a
      ``pd.DataFrame``. A panel is transformed to a ``pd.DataFrame``, with
      as many rows as time series in the panel.
      A hierarchical series is transformed to a ``pd.DataFrame`` with the hierarchy
      indices retained, one row corresponding to a non-temporal leaf node.
      Examples are feature extraction or summarization (mean, quantiles, etc), see
      ``SummaryTransformer``.
    * Series-to-Panel, this transforms individual series to a panel of time series.
      Panels are transformed to hierarchical series with added index levels.
      Examples are time series bootstrapping, where multiple bootstrap samples are
      produced per input series, see ``STLBootstrapTransformer``,
      or ``TSBootstrapAdapter``.
    * Panel-to-Series, this transforms a panel of time series to a single time series.
      Examples are aggregation with time index retained, e.g., mean per index or bin,
      see ``Merger``.

    The relationship between input and output types of ``transform`` is
    summarized in the following table,
    for the case where ``"scitype:transform-input"`` is ``"Series"``.

    The first column is the type of ``X``, which need not be ``"Series"``,
    the second column is the value of the ``"scitype:transform-output"`` tag,
    the third column is the type of the output of ``transform``.

    The output type is obtained from the input type of ``transform``, from
    broadcasting of the types defined by the tag values.

    .. list-table::
        :widths: 35 35 40
        :header-rows: 2

        * -
            - `transform`
            -
        * - `X`
            - `-output`
            - type of return
        * - `Series`
            - `Primitives`
            - `pd.DataFrame` (1-row)
        * - `Panel`
            - `Primitives`
            - `pd.DataFrame`
        * - `Series`
            - `Series`
            - `Series`
        * - `Panel`
            - `Series`
            - `Panel`
        * - `Series`
            - `Panel`
            - `Panel`

    The instance indices in the in return correspond to instances in the input ``X``.
    """

    _tags = {
        "tag_name": "scitype:transform-output",
        "parent_type": "transformer",
        "tag_type": ("str", ["Series", "Panel", "Primitives"]),
        "short_descr": "what is the scitype of the transformer output, the transformed X",  # noqa: E501
        "user_facing": True,
    }


class requires_x(_BaseTag):
    """Behaviour flag: transformer requires X in fit and transform.

    - String name: ``"requires_X"``
    - Public behaviour flag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``True``

    This tag applies to transformations.

    If the tag is ``True``, the transformer requires the input data argument ``X``
    to be passed in both the ``fit`` and ``transform`` methods, as well as in
    other methods that require input data, if available.

    If the tag is ``False``, the transformer does not require the
    input data argument ``X`` to be passed in any method.
    """

    _tags = {
        "tag_name": "requires_X",
        "parent_type": "transformer",
        "tag_type": "bool",
        "short_descr": "does the transformer require X to be passed in fit and transform?",  # noqa: E501
        "user_facing": True,
    }


class requires_y(_BaseTag):
    """Behaviour flag: transformer requires y in fit.

    - String name: ``"requires_y"``
    - Public behaviour flag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    This tag applies to transformations.

    If the tag is ``True``, the transformer requires the target data argument ``y``
    to be passed in the ``fit`` method, as well as in the ``update`` method.
    The type of ``y`` required is specified by the tag ``scitype:transform-labels``.
    The requirement to pass ``y`` is usually in addition to passing ``X``.

    If the tag is ``True``, it does not necessarily imply that ``y`` is also
    required in the ``transform`` or ``inverse_transform`` methods. This may
    be the case, but is not implied by this tag.
    Usually, ``y`` is not required in ``transform`` or ``inverse_transform``.
    There is currently no tag to specify this requirement, users should
    consult the documentation of the transformer.

    If the tag is ``False``, the transformer does not require the
    target data argument ``y`` to be passed in any method.
    """

    _tags = {
        "tag_name": "requires_y",
        "parent_type": "transformer",
        "tag_type": "bool",
        "short_descr": "does the transformer require y to be passed in fit and transform?",  # noqa: E501
        "user_facing": True,
    }


class scitype__transform_labels(_BaseTag):
    """The scitype of the target data for the transformer, if required.

    - String name: ``"scitype:transform-labels"``
    - Public scitype tag
    - Values: string, one of ``"None"``, ``"Series"``, ``"Primitives"``, ``"Panel"``
    - Example: ``"Series"``
    - Default: ``"None"``
    - Alias: ``"scitype:y"``

    This tag applies to transformations.

    The tag specifies the scitype of the target data ``y`` that is required,
    in a case where the transformer requires target data, i.e., the
    tag ``requires_y`` is ``True``.

    The possible values are:

    * ``"None"``: no target data is required. This value is used if and only if
      the transformer does not require target data, i.e., the tag ``requires_y``
      is ``False``.
    * ``"Series"``: a single time series, in ``Series`` :term:`scitype`.
      If the tag ``X-y-must-have-same-index`` is ``True``, then the index, or implied
      index, of the target series must be the same as the index of the
      input series ``X``.
    * ``"Primitives"``: a collection of primitive types, e.g., a collection of scalars,
      in ``Table`` :term:`scitype`. In this case, the number of rows (=instances)
      in ``y`` must always equal the number of instances in ``X``, which typically
      will be of :mtype:`scitype` ``Panel`` in this case.
    * ``"Panel"``: a panel of time series, in ``Panel`` :term:`scitype`.

    The tag ``scitype:transform-labels`` is used in conjunction with the tag
    ``requires_y``, which specifies whether target data is required by the transformer.

    If the tag ``requires_y`` is ``False``, then the tag ``scitype:transform-labels``
    will be ``"None"``.
    """

    _tags = {
        "tag_name": "scitype:transform-labels",
        "parent_type": "transformer",
        "tag_type": ("str", ["None", "Series", "Primitives", "Panel"]),
        "short_descr": "what is the scitype of the target labels y, if required?",
        "user_facing": True,
    }


class capability__inverse_transform(_BaseTag):
    """Capability: the transformer can carry out an inverse transform.

    - String name: ``"capability:inverse_transform"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``
    - Alias: ``"inverse-transform"``

    This tag applies to transformations.

    If the tag is ``True``, the transformer can carry out an inverse transform,
    i.e., the transformer can carry out the operation that is an inverse,
    pseudo-inverse, or approximate inverse (such as denoising inverse) of
    the operation carried out by the ``transform`` method.

    The inverse transform is available via the method ``inverse_transform``.

    If ``inverse_transform`` is available, the
    following tags specify additional properties and behaviour of the inverse transform:

    * ``"capability:inverse_transform:range"``: the domain of invertibility of
      the transform.
    * ``"capability:inverse_transform:exact"``: whether the inverse transform is
      expected to be an exact inverse to the transform.
    * ``"skip-inverse-transform"``: if used in a pipeline, the transformer will
      skip the inverse transform, if the tag is ``True``.

    If the ``capability:inverse_transform`` tag is ``False``,
    the transformer cannot carry out an inverse transform,
    and will raise an error if an inverse transform is attempted.
    """

    _tags = {
        "tag_name": "capability:inverse_transform",
        "parent_type": "transformer",
        "tag_type": "bool",
        "short_descr": "is the transformer capable of carrying out an inverse transform?",  # noqa: E501
        "user_facing": True,
    }


class capability__inverse_transform__range(_BaseTag):
    """Capability: the domain of invertibility of the transform.

    - String name: ``"capability:inverse_transform:range"``
    - Public capability tag
    - Values: list, [lower, upper], of float
    - Example: [0.0, 1.0]
    - Default: ``None``

    This tag applies to transformations that possess an ``inverse_transform`` method,
    as specified by the tag ``capability:inverse_transform``.
    It is one of the tags that specify the properties of the inverse transform.

    The tag specifies the domain of invertibility of the transform, i.e.,
    the range of values for which the inverse transform is mathematically defined.

    This is the same as the subset of the domain of the transform for which
    the transform is invertible. In general, the domain of invertibility
    will be smaller than the domain of the transform, but may be equal.

    The tag value is a list of two floats, [lower, upper], where:

    * ``lower``: the lower bound of the domain of invertibility.
    * ``upper``: the upper bound of the domain of invertibility.

    These two values may depend on hyper-parameters of the transformer,
    as well as the data seen in the ``fit`` method.

    If the tag value is ``None``, the domain of invertibility is assumed to be
    the entire domain of the transform.

    If ``"capability:inverse_transform"`` is ``False``, this tag is irrelevant
    and will also have value ``None``.
    """

    _tags = {
        "tag_name": "capability:inverse_transform:range",
        "parent_type": "transformer",
        "tag_type": "list",
        "short_descr": "domain of invertibility of transform, must be list [lower, upper] of float",  # noqa: E501
        "user_facing": True,
    }


class capability__inverse_transform__exact(_BaseTag):
    """Capability: whether the inverse transform is an exact inverse to the transform.

    - String name: ``"capability:inverse_transform:exact"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    This tag applies to transformations that possess an ``inverse_transform`` method,
    as specified by the tag ``capability:inverse_transform``.
    It is one of the tags that specify the properties of the inverse transform.

    The tag specifies whether the inverse transform is expected to be an exact inverse
    to the transform, i.e., whether the inverse transform is mathematically defined
    as the exact inverse of the transform.

    If the tag is ``True``, applying ``inverse_transform`` to the
    output of ``transform`` should yield the original input data,
    up to numerical precision.

    If the tag is ``False``, the inverse transform is not expected to be an exact
    inverse of the transform, and may be an approximate inverse, pseudo-inverse,
    or denoising inverse.
    While there is a general expectation that the inverse transform should be
    close to a reasonable inverse, if it is well-defined,
    this is not a strict requirement of the interface.
    """

    _tags = {
        "tag_name": "capability:inverse_transform:exact",
        "parent_type": "transformer",
        "tag_type": "bool",
        "short_descr": "whether inverse_transform is expected to be an exact inverse to transform",  # noqa: E501
        "user_facing": True,
    }


class transform_returns_same_time_index(_BaseTag):
    """Property: transformer returns same time index as input.

    - String name: ``"transform-returns-same-time-index"``
    - Public property tag
    - Values: boolean, ``True`` / ``False``
    - Example: ``True``
    - Default: ``False``

    This tag applies to transformations.

    If the tag is ``True``, the transformer returns a transformed series
    with the same time index as the input series ``X``.

    This tag applies only to transformers that return time series as output, i.e.,
    the tag ``scitype:transform-output`` is ``"Series"`` or ``"Panel"``.

    In cases where input and output :term:`mtype` do not have explicit time index,
    the tag applies to the implicit time index, i.e., the index of the abstract series
    representation, for instance, integer index in case of ``numpy`` arrays,
    in which case the implication is that an an array is returned with
    equal length in the dimension corresponding to the time index.

    If the tag is ``False``, the returned series will in general have a different
    time index than the input series.

    If ``scitype:transform-output`` is ``"Primitives"``, this tag is irrelevant
    and will have value ``False``.

    Besides being informative to the user, this tag is also used internally
    by the framework to track guarantees on the data index.
    """

    _tags = {
        "tag_name": "transform-returns-same-time-index",
        "parent_type": "transformer",
        "tag_type": "bool",
        "short_descr": "does transform return same time index as input?",
        "user_facing": True,
    }


# Developer tags
# --------------


class x_inner_mtype(_BaseTag):
    """The machine type(s) the transformer can deal with internally for X.

    - String name: ``"X_inner_mtype"``
    - Extension developer tag
    - Values: str or list of string, from the list of :term:`mtype` strings
    - Example: ``"pd.DataFrame"``
    - Default: specific to estimator type, see extension template

    Estimators in ``sktime`` support a variety of input data types, following
    one of many possible machine types, short: :term:`mtype` specifications.

    Internally, the estimator may support only a subset of these types,
    for instance due to the implementation of the estimator, or due to
    interfacing with external libraries that use a specific data format.

    The ``sktime`` extension contracts allow the extender to specify the
    internal :term:`mtype` support, in this case the methods the extender
    needs to implement guarantee that the arguments are of the correct type,
    by carrying out the necessary conversions and coercions.

    For instance, an extender implementing ``_fit`` with an ``X`` argument
    and ``X_inner_mtype`` set to ``"pd.DataFrame"`` can assume that the ``X``
    argument follows the ``pd.DataFrame`` :term:`mtype` specification - while
    users can pass any supported mtype to the public ``fit`` method.

    Tags named ``X_inner_mtype``, ``y_inner_mtype``, etc, apply this
    specification to the respective arguments in the method signature.

    The four main patterns in using the "inner mtype" tag are as follows:

    * specifying a single string. In this case, internal methods will provide
      the extender with inputs in the specified machine type.
    * specifying a list of strings, of the same :mtype:`scitype`.
      In this case, the boilerplate layer will
      first attempt to find the first :term:`mtype` in the list.
    * specifying a list of strings, all of different :mtype:`scitype`.
      This will convert the input to the mtype of the same scitype. This is especially
      useful if the implementer wants to deal with scitype broadcasting internally,
      in this case it is recommended to specify similar mtypes, such as
      ``"pd.DataFrame"``, ``"pd-multiindex"``, ``"pd_multiindex_hier``,
      which allow dealing with the different types simultaneously.
    * specifying all possible mtypes, by setting the default to a list such as
      ``ALL_TIME_SERIES_MTYPES`` from the ``datatypes`` module.
      As all mtypes are supported, inputs will be passed through to ``_fit`` etc,
      without any conversion and coercion. This is useful for composites,
      where the extender wants to ensure that components should carry out
      the necessary conversions and coercions.

    More generally, for an arbitrary list of mtypes, the boilerplate logic will:

    * first checks whether the mtype of the input is on the list. If yes,
      the input will be passed through as is.
    * if the mtype of the input is not on the list, the boilerplate will attempt to
      identify the first mtype of the same scitype as the input, and coerce to that.
    * if no mtype of same scitype is found, it will attempt to coerce to the
      "simplest" adjacent scitype, e.g., from ``"pd.DataFrame"`` to ``"pd-multiindex"``.

    In all cases, ordering is important, as the first mtype in the list is the
    one that will be used as target type for conversions.
    """

    _tags = {
        "tag_name": "X_inner_mtype",
        "parent_type": "estimator",
        "tag_type": ("list", "str"),
        "short_descr": "which machine type(s) is the internal _fit/_predict able to deal with?",  # noqa: E501
        "user_facing": False,
    }


class y_inner_mtype(_BaseTag):
    """The machine type(s) the transformer can deal with internally for y.

    - String name: ``"y_inner_mtype"``
    - Extension developer tag
    - Values: str or list of string, from the list of :term:`mtype` strings
    - Example: ``"pd.DataFrame"``
    - Default: specific to estimator type, see extension template

    Estimators in ``sktime`` support a variety of input data types, following
    one of many possible machine types, short: :term:`mtype` specifications.

    Internally, the estimator may support only a subset of these types,
    for instance due to the implementation of the estimator, or due to
    interfacing with external libraries that use a specific data format.

    The ``sktime`` extension contracts allow the extender to specify the
    internal :term:`mtype` support, in this case the methods the extender
    needs to implement guarantee that the arguments are of the correct type,
    by carrying out the necessary conversions and coercions.

    For instance, an extender implementing ``_fit`` with an ``X`` argument
    and ``X_inner_mtype`` set to ``"pd.DataFrame"`` can assume that the ``X``
    argument follows the ``pd.DataFrame`` :term:`mtype` specification - while
    users can pass any supported mtype to the public ``fit`` method.

    Tags named ``X_inner_mtype``, ``y_inner_mtype``, etc, apply this
    specification to the respective arguments in the method signature.

    The four main patterns in using the "inner mtype" tag are as follows:

    * specifying a single string. In this case, internal methods will provide
      the extender with inputs in the specified machine type.
    * specifying a list of strings, of the same :mtype:`scitype`.
      In this case, the boilerplate layer will
      first attempt to find the first :term:`mtype` in the list.
    * specifying a list of strings, all of different :mtype:`scitype`.
      This will convert the input to the mtype of the same scitype. This is especially
      useful if the implementer wants to deal with scitype broadcasting internally,
      in this case it is recommended to specify similar mtypes, such as
      ``"pd.DataFrame"``, ``"pd-multiindex"``, ``"pd_multiindex_hier``,
      which allow dealing with the different types simultaneously.
    * specifying all possible mtypes, by setting the default to a list such as
      ``ALL_TIME_SERIES_MTYPES`` from the ``datatypes`` module.
      As all mtypes are supported, inputs will be passed through to ``_fit`` etc,
      without any conversion and coercion. This is useful for composites,
      where the extender wants to ensure that components should carry out
      the necessary conversions and coercions.

    More generally, for an arbitrary list of mtypes, the boilerplate logic will:

    * first checks whether the mtype of the input is on the list. If yes,
      the input will be passed through as is.
    * if the mtype of the input is not on the list, the boilerplate will attempt to
      identify the first mtype of the same scitype as the input, and coerce to that.
    * if no mtype of same scitype is found, it will attempt to coerce to the
      "simplest" adjacent scitype, e.g., from ``"pd.DataFrame"`` to ``"pd-multiindex"``.

    In all cases, ordering is important, as the first mtype in the list is the
    one that will be used as target type for conversions.
    """

    _tags = {
        "tag_name": "y_inner_mtype",
        "parent_type": "estimator",
        "tag_type": ("list", "str"),
        "short_descr": "which machine type(s) is the internal _fit/_predict able to deal with?",  # noqa: E501
        "user_facing": False,
    }


ESTIMATOR_TAG_REGISTER = [
    (
        "skip-inverse-transform",
        "transformer",
        "bool",
        "behaviour flag: skips inverse_transform when called yes/no",
    ),
    (
        "X-y-must-have-same-index",
        ["forecaster", "regressor", "transformer"],
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
        "pwtrafo_type",
        ["transformer-pairwise", "transformer-pairwise-panel"],
        ("str", ["distance", "kernel", "other"]),
        "mathematical type of pairwise transformer - distance, kernel, or other",
    ),
    (
        "scitype:X",
        "param_est",
        "str",
        "which scitypes does X internally support?",
    ),
    (
        "scitype:instancewise",
        "transformer",
        "bool",
        "does the transformer transform instances independently?",
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
        "alignment_type",
        "aligner",
        ("str", ["full", "partial"]),
        "does aligner produce a full or partial alignment",
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
        "remember_data",
        ["forecaster", "transformer"],
        "bool",
        "whether estimator remembers all data seen as self._X, self._y, etc",
    ),
    (
        "distribution_type",
        "estimator",
        "str",
        "distribution type of data as str",
    ),
    (
        "task",
        "series-annotator",
        "str",
        "subtype of series annotator, e.g., 'anomaly_detection', 'segmentation'",
    ),
    (
        "learning_type",
        "series-annotator",
        "str",
        "type of learning, e.g., 'supervised', 'unsupervised'",
    ),
    (
        "reserved_params",
        "estimator",
        ("list", "str"),
        "parameters reserved by the base class and present in all child estimators",
    ),
    (
        "split_hierarchical",
        "splitter",
        "bool",
        "whether _split is natively implemented for hierarchical y types",
    ),
    (
        "split_series_uses",
        "splitter",
        ("str", ["iloc", "loc", "custom"]),
        "whether split_series uses split (iloc) or split_loc (loc) to split series",
    ),
    (
        "split_type",
        "splitter",
        ("str", ["temporal", "instance"]),
        "whether the splitter splits by time or by instance (panel/hierarchy index)",
    ),
    # -------------------------
    # tags to be moved to skpro
    # -------------------------
    # these tags will be moved to skpro
    # some to be converted to configs, see skpro issue #269
    (
        "distribution_type",
        "estimator",
        "str",
        "distribution type of data as str",
    ),
    (
        "capabilities:exact",
        "distribution",
        ("list", "str"),
        "methods provided by the distribution that return numerically exact results",
    ),
    (
        "capabilities:approx",
        "distribution",
        ("list", "str"),
        "methods provided by the distribution that return approximate results",
    ),
    (
        "distr:measuretype",
        "distribution",
        ("str", ["continuous", "discrete", "mixed"]),
        "class the distribution measure belongs to - abs.continuous, discrete, mixed",
    ),
    (
        "approx_mean_spl",
        "distribution",
        "int",
        "sample size used in approximating generative mean if not available",
    ),
    (
        "approx_var_spl",
        "distribution",
        "int",
        "sample size used in approximating generative variance if not available",
    ),
    (
        "approx_energy_spl",
        "distribution",
        "int",
        "sample size used in approximating generative energy if not available",
    ),
    (
        "approx_spl",
        "distribution",
        "int",
        "sample size used in approximating other statistics if not available",
    ),
    (
        "bisect_iter",
        "distribution",
        "int",
        "max iters for bisection method in ppf",
    ),
    # ---------------------
    # to be renamed/aliased
    # ---------------------
    # the following tags are to be renamed or aliased
    (
        "univariate-only",  # -> capability:multivariate, invert
        "transformer",
        "bool",
        "can transformer handle multivariate series? True = no",
    ),
    (
        "univariate-metric",  # -> capability:multivariate, invert
        "metric",
        "bool",
        "Does the metric only work on univariate y data?",
    ),
    (
        "handles-missing-data",  # -> capability:missing_values
        "estimator",
        "bool",
        "can the estimator handle missing data (NA, np.nan) in inputs?",
    ),
    (
        "scitype:y",  # -> capability:multivariate
        # the scitype:y tag should be kept but for separate use,
        # a list of the internal scitypes supported by the estimator
        # or the base scitype of the target data
        "forecaster",
        ("str", ["univariate", "multivariate", "both"]),
        "which series type does the forecaster support? multivariate means >1 vars",
    ),
    # ---------------------------
    # to be deprecated or removed
    # ---------------------------
    # the following tags are to be deprecated or removed
    (
        "capability:pred_var",  # redundant with capability:pred_int
        # because if one of the proba methods is available, all others are too
        "forecaster",
        "bool",
        "does the forecaster implement predict_variance?",
    ),
    (
        "capability:global_forecasting",
        ["forecaster"],
        "bool",
        "can the estimator make global forecasting?",
    ),
    (
        "python_dependencies_alias",
        "object",
        "dict",
        "deprecated tag for dependency import aliases",
    ),
]

# construct the tag register from all classes in this module
tag_clses = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for _, cl in tag_clses:
    # skip the base class
    if cl.__name__ == "_BaseTag" or not issubclass(cl, _BaseTag):
        continue

    cl_tags = cl.get_class_tags()

    tag_name = cl_tags["tag_name"]
    parent_type = cl_tags["parent_type"]
    tag_type = cl_tags["tag_type"]
    short_descr = cl_tags["short_descr"]

    ESTIMATOR_TAG_REGISTER.append((tag_name, parent_type, tag_type, short_descr))

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
