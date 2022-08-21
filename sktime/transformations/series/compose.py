#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Meta-transformers for building composite transformers."""

__author__ = ["aiwalter", "SveaMeyer13", "fkiraly"]
__all__ = ["OptionalPassthrough", "ColumnwiseTransformer", "YtoX"]

from warnings import warn

from sktime.transformations.compose import (
    ColumnwiseTransformer,
    OptionalPassthrough,
    YtoX,
)

warn(
    "transformations.series.compose is deprecated and will be removed in"
    " version 0.15.0. All estimators in it will be moved to transformations.compose. "
    "Please change imports to transformations.compose to avoid breakage.",
    DeprecationWarning,
)
