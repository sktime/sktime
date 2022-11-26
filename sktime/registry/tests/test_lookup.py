# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Testing of registry lookup functionality."""

__author__ = ["fkiraly"]

import pytest

from sktime.base import BaseObject
from sktime.registry import all_estimators, all_tags, scitype
from sktime.registry._base_classes import (
    BASE_CLASS_LOOKUP,
    BASE_CLASS_SCITYPE_LIST,
    TRANSFORMER_MIXIN_SCITYPE_LIST,
)
from sktime.registry._lookup import _check_estimator_types

VALID_SCITYPES_SET = set(
    BASE_CLASS_SCITYPE_LIST + TRANSFORMER_MIXIN_SCITYPE_LIST + ["estimator"]
)

# some scitypes have no associated tags yet
SCITYPES_WITHOUT_TAGS = [
    "series-annotator",
    "clusterer",
    "object",
    "splitter",
    "network",
]

# shorthands for easy reading
b = BASE_CLASS_SCITYPE_LIST
n = len(b)

# selected examples of "search for two types at once to avoid quadratic scaling"
double_estimator_scitypes = [[b[i], b[(i + 3) % n]] for i in range(n)]
# fixtures search by individual scitypes, "None", and some pairs
estimator_scitype_fixture = [None] + BASE_CLASS_SCITYPE_LIST + double_estimator_scitypes


def _to_list(obj):
    """Put obj in list if it is not a list."""
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


def _get_type_tuple(estimator_scitype):
    """Convert scitype string(s) into tuple of classes for isinstance check.

    Parameters
    ----------
    estimator_scitypes: None, string, or list of string

    Returns
    -------
    estimator_classes : tuple of sktime base classes,
        corresponding to scitype strings in estimator_scitypes
    """
    if estimator_scitype is not None:

        estimator_classes = tuple(
            BASE_CLASS_LOOKUP[scitype] for scitype in _to_list(estimator_scitype)
        )
    else:
        estimator_classes = (BaseObject,)

    return estimator_classes


@pytest.mark.parametrize("return_names", [True, False])
@pytest.mark.parametrize("estimator_scitype", estimator_scitype_fixture)
def test_all_estimators_by_scitype(estimator_scitype, return_names):
    """Check that all_estimators return argument has correct type."""
    estimators = all_estimators(
        estimator_types=estimator_scitype,
        return_names=return_names,
    )

    estimator_classes = _get_type_tuple(estimator_scitype)

    assert isinstance(estimators, list)
    # there should be at least one estimator returned
    assert len(estimators) > 0

    # checks return type specification (see docstring)
    if return_names:
        for estimator in estimators:
            assert isinstance(estimator, tuple) and len(estimator) == 2
            assert isinstance(estimator[0], str)
            assert issubclass(estimator[1], estimator_classes)
            assert estimator[0] == estimator[1].__name__
    else:
        for estimator in estimators:
            assert issubclass(estimator, estimator_classes)


@pytest.mark.parametrize("estimator_scitype", estimator_scitype_fixture)
def test_all_tags(estimator_scitype):
    """Check that all_tags return argument has correct type."""
    tags = all_tags(estimator_types=estimator_scitype)
    assert isinstance(tags, list)

    # there should be at least one tag returned
    # exception: scitypes which we know don't have tags associated
    est_list = (
        estimator_scitype
        if isinstance(estimator_scitype, list)
        else [estimator_scitype]
    )
    if not set(est_list).issubset(SCITYPES_WITHOUT_TAGS):
        assert len(tags) > 0

    # checks return type specification (see docstring)
    for tag in tags:
        assert isinstance(tag, tuple)
        assert isinstance(tag[0], str)
        assert VALID_SCITYPES_SET.issuperset(_to_list(tag[1]))
        assert isinstance(tag[2], (str, tuple))
        if isinstance(tag[2], tuple):
            assert len(tag[2]) == 2
            assert isinstance(tag[2][0], str)
            assert isinstance(tag[2][1], (str, list))
        assert isinstance(tag[3], str)


@pytest.mark.parametrize("return_names", [True, False])
def test_all_estimators_return_names(return_names):
    """Test return_names argument in all_estimators."""
    estimators = all_estimators(return_names=return_names)
    assert isinstance(estimators, list)
    assert len(estimators) > 0

    if return_names:
        assert all([isinstance(estimator, tuple) for estimator in estimators])
        names, estimators = list(zip(*estimators))
        assert all([isinstance(name, str) for name in names])
        assert all(
            [name == estimator.__name__ for name, estimator in zip(names, estimators)]
        )

    assert all([isinstance(estimator, type) for estimator in estimators])


# arbitrary list for exclude_estimators argument test
EXCLUDE_ESTIMATORS = [
    "ElasticEnsemble",
    "ProximityForest",
    "ProximityStump",
    "ProximityTree",
]


@pytest.mark.parametrize("exclude_estimators", ["NaiveForecaster", EXCLUDE_ESTIMATORS])
def test_all_estimators_exclude_estimators(exclude_estimators):
    """Test exclued_estimators argument in all_estimators."""
    estimators = all_estimators(
        return_names=True, exclude_estimators=exclude_estimators
    )
    assert isinstance(estimators, list)
    assert len(estimators) > 0
    names, estimators = list(zip(*estimators))

    if not isinstance(exclude_estimators, list):
        exclude_estimators = [exclude_estimators]
    for estimator in exclude_estimators:
        assert estimator not in names


def _get_tag_fixture():
    """Generate a simple list of test cases for optional return_tags."""
    # just picked a few valid tags to try out as valid str return_tags args:
    test_str_as_arg = [
        "X-y-must-have-same-index",
        "capability:pred_var",
        "skip-inverse-transform",
    ]

    # we can also make them into a list to test list of str as a valid arg:
    test_list_as_arg = [test_str_as_arg]
    # Note - I don't include None explicitly as a test case - tested elsewhere
    return test_str_as_arg + test_list_as_arg


# test that all_estimators returns as expected if given correct return_tags:
@pytest.mark.parametrize("return_tags", _get_tag_fixture())
@pytest.mark.parametrize("return_names", [True, False])
def test_all_estimators_return_tags(return_tags, return_names):
    """Test ability to return estimator value of passed tags."""
    estimators = all_estimators(
        return_tags=return_tags,
        return_names=return_names,
    )
    # Helps us keep track of estimator index within the tuple:
    ESTIMATOR_INDEX = 1 if return_names else 0
    TAG_START_INDEX = ESTIMATOR_INDEX + 1

    assert isinstance(estimators[0], tuple)
    # check length of tuple is what we expect:
    if isinstance(return_tags, str):
        assert len(estimators[0]) == TAG_START_INDEX + 1
    else:
        assert len(estimators[0]) == len(return_tags) + TAG_START_INDEX

    # check that for each estimator the value for that tag is correct:
    for est_tuple in estimators:
        est = est_tuple[ESTIMATOR_INDEX]
        if isinstance(return_tags, str):
            assert est.get_class_tag(return_tags) == est_tuple[TAG_START_INDEX]
        else:
            for tag_index, tag in enumerate(return_tags):
                assert est.get_class_tag(tag) == est_tuple[TAG_START_INDEX + tag_index]


def _get_bad_return_tags():
    """Get return_tags arguments that should throw an exception."""
    # case not a str or a list:
    is_int = [12]
    # case is a list, but not all elements are str:
    is_not_all_str = [["this", "is", "a", "test", 12, "!"]]

    return is_int + is_not_all_str


# test that all_estimators breaks as expected if given bad return_tags:
@pytest.mark.parametrize("return_tags", _get_bad_return_tags())
def test_all_estimators_return_tags_bad_arg(return_tags):
    """Test ability to catch bad arguments of return_tags."""
    with pytest.raises(TypeError):
        _ = all_estimators(return_tags=return_tags)


@pytest.mark.parametrize("estimator_scitype", BASE_CLASS_SCITYPE_LIST)
def test_scitype_inference(estimator_scitype):
    """Check that scitype inverts _check_estimator_types."""
    base_class = _check_estimator_types(estimator_scitype)[0]
    inferred_scitype = scitype(base_class)

    assert (
        inferred_scitype == estimator_scitype
    ), "one of scitype, _check_estimator_types is incorrect, these should be inverses"
