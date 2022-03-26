# -*- coding: utf-8 -*-
"""Feature union pipeline element."""

from warnings import warn

from sktime.transformations.compose import FeatureUnion

__all__ = ["FeatureUnion"]

warn(
    "FeatureUnion has been moved to transformations.compose,"
    " the old location in series_as_features is deprecated since 0.11.0,"
    " and will be removed in 0.12.0. Please use the import from "
    "transformations.compose import FeatureUnion."
)
