# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Isolated pandas conversion layer for ForecastingHorizonV2.

ALL pandas-specific imports and logic live in this module. The core
_FHValues and ForecastingHorizonV2 classes should never import pandas
directly — they go through this converter instead.

This module handles:
- Converting user-facing input types (int, list, pd.Index, etc.)
  to the internal _FHValues representation.
- Converting _FHValues back to pd.Index for interop with sktime.
- Extracting and normalizing frequency strings from pandas objects.
- Converting cutoff values from pandas types to _FHValues.
"""
