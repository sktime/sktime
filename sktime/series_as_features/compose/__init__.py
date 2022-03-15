#!/usr/bin/env python3 -u
"""Deprecated - init for FeatureUnion module."""
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["FeatureUnion"]

from warnings import warn

from sktime.transformations.compose import FeatureUnion

warn(
    "FeatureUnion has been moved to sktime.transformations.compose in 0.11.0. "
    "This location (series_as_features) is deprecated and will be removed in 0.12.0. "
)
