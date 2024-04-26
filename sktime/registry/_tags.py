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
    - ``"python_dependencies_alias"``: alias for package names,
      if different from import names
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
    - ``"python_dependencies_alias"``: alias for package names,
      if different from import names
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


class python_dependencies_alias(_BaseTag):
    """Alias for Python package dependency names for the object.

    Part of packaging metadata for the object.

    - String name: ``"python_dependencies_alias"``
    - Private tag, developer and framework facing
    - Values: dict of str, key = PEP 440 package name, value = import name
    - Example: ``{"scikit-learn": "sklearn"}``
    - Example 2: ``{"dtw-python": "dtw", "scikit-learn": "sklearn"}``
    - Default: no aliases (``None``)

    ``sktime`` manages objects and estimators like mini-packages,
    with their own dependencies and compatibility requirements.
    Dependencies are specified in the tags:

    - ``"python_version"``: Python version specifier (PEP 440) for the object,
    - ``"python_dependencies"``: list of required Python packages (PEP 440)
    - ``"python_dependencies_alias"``: alias for package names,
      if different from import names
    - ``"env_marker"``: environment marker for the object (PEP 508)
    - ``"requires_cython"``: whether the object requires a C compiler present

    The ``python_dependencies_alias`` tag of an object is dict,
    providing import name aliases for package names in the ``python_dependencies`` tag,
    if the package name differs from the import name.

    The tag is used in packaging metadata for the object,
    and is used internally to check compatibility of the object with
    the build environment, to raise informative error messages.

    This tag is required if the package name of a dependency is different
    from the import name of the package, e.g., ``"dtw-python"`` and ``"dtw"``.
    If not set, the package name is assumed to be identical with the import name.

    Developers should note that elements of this ``dict`` are not passed on
    via field inheritance, unlike the tags themselves.
    Hence, if multiple aliases are required, they need to be set in the same tag.
    """

    _tags = {
        "tag_name": "python_dependencies_alias",
        "parent_type": "object",
        "tag_type": "dict",
        "short_descr": "alias for package names in python_dependencies, key-value pairs are package name, import name",  # noqa: E501
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
    - ``"python_dependencies_alias"``: alias for package names,
      if different from import names
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
    - ``"python_dependencies_alias"``: alias for package names,
      if different from import names
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

    Feature importances are queriable by the fitted parameter interface
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

    Training performance estimates are queriable by the fitted parameter interface
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
    the ``X`` parmameter in ``fit``, ``predict``, and other methods
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


ESTIMATOR_TAG_REGISTER = [
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
                "np.ndarray",
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
        [
            "clusterer",
            "forecaster",
            "transformer",
            "transformer-pairwise-panel",
            "param_est",
        ],
        (
            "list",
            [
                "pd.Series",
                "pd.DataFrame",
                "np.ndarray",
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
        "requires_X",
        "transformer",
        "bool",
        "does this transformer require X to be passed in fit and transform?",
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
        "capability:inverse_transform:range",
        "transformer",
        "list",
        "domain of invertibility of transform, must be list [lower, upper] of float",
    ),
    (
        "capability:inverse_transform:exact",
        "transformer",
        "bool",
        "whether inverse_transform is expected to be an exact inverse to transform",
    ),
    (
        "capability:pred_var",
        "forecaster",
        "bool",
        "does the forecaster implement predict_variance?",
    ),
    (
        "capability:predict_proba",
        "classifier",
        "bool",
        "does the classifier implement a non-default predict_proba, "
        "i.e., not just 0/1 probabilities obtained from predict?",
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
    (
        "capability:multioutput",
        ["classifier", "regressor"],  # might need to add "early_classifier" here
        "bool",
        "can the estimator handle multioutput data?",
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
