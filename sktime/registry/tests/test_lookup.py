# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Testing of registry lookup functionality."""

__author__ = ["fkiraly"]

import pytest

from sktime.base import BaseEstimator
from sktime.registry import all_estimators, all_tags
from sktime.registry._base_classes import (
    BASE_CLASS_LOOKUP,
    BASE_CLASS_SCITYPE_LIST,
    TRANSFORMER_MIXIN_SCITYPE_LIST,
)

VALID_SCITYPES_SET = set(
    BASE_CLASS_SCITYPE_LIST + TRANSFORMER_MIXIN_SCITYPE_LIST + ["estimator"]
)

double_estimator_scitypes = [
    [x, y] for x in BASE_CLASS_SCITYPE_LIST for y in BASE_CLASS_SCITYPE_LIST
]
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
        estimator_classes = BaseEstimator,

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

    for tag in tags:
        assert isinstance(tag, tuple)
        assert isinstance(tag[0], str)
        print(VALID_SCITYPES_SET)
        assert VALID_SCITYPES_SET.issuperset(_to_list(tag[1]))
        assert isinstance(tag[2], (str, tuple))
        if isinstance(tag[2], tuple):
            assert len(tag[2]) == 2
            assert isinstance(tag[2][0], str)
            assert isinstance(tag[2][1], list)
        assert isinstance(tag[3], str)
