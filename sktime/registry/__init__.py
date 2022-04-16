# -*- coding: utf-8 -*-
"""Implements registry for sktime estimator base classes and tags."""

from sktime.registry._base_classes import (
    BASE_CLASS_LIST,
    BASE_CLASS_LOOKUP,
    BASE_CLASS_REGISTER,
    BASE_CLASS_SCITYPE_LIST,
    TRANSFORMER_MIXIN_LIST,
    TRANSFORMER_MIXIN_LOOKUP,
    TRANSFORMER_MIXIN_REGISTER,
    TRANSFORMER_MIXIN_SCITYPE_LIST,
)
from sktime.registry._lookup import all_estimators, all_tags
from sktime.registry._scitype import scitype
from sktime.registry._tags import (
    ESTIMATOR_TAG_LIST,
    ESTIMATOR_TAG_REGISTER,
    check_tag_is_valid,
)

__all__ = [
    "all_estimators",
    "all_tags",
    "check_tag_is_valid",
    "scitype",
    "ESTIMATOR_TAG_LIST",
    "ESTIMATOR_TAG_REGISTER",
    "BASE_CLASS_REGISTER",
    "BASE_CLASS_LIST",
    "BASE_CLASS_LOOKUP",
    "BASE_CLASS_SCITYPE_LIST",
    "TRANSFORMER_MIXIN_REGISTER",
    "TRANSFORMER_MIXIN_LIST",
    "TRANSFORMER_MIXIN_LOOKUP",
    "TRANSFORMER_MIXIN_SCITYPE_LIST",
]
