# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Isolated pandas conversion layer for ForecastingHorizonV2.

ALL pandas-specific imports and logic live in this module.
The core _FHValues and ForecastingHorizonV2 classes should never import pandas directly,
they go through this converter instead.

This module handles:
Converting user-facing input types (int, list, pd.Index, etc.)
to the internal _FHValues representation.
Converting _FHValues back to pd.Index for interoperability with sktime.
Extracting and normalizing frequency strings from pandas objects.
Converting cutoff values from pandas types to _FHValues.
"""

import pandas as pd

from sktime.forecasting.base._fh_values import FHValues


class PandasFHConverter:
    """Static conversion layer between pandas types and FHValues.

    This class collects all pandas-coupled logic in one place so that
    the rest of the ForecastingHorizonV2 code can remain pandas-free.
    """

    # input -> FHValues (internal representation) conversion
    @staticmethod
    def to_internal():
        pass

    # FHValues (internal representation) -> pandas conversion
    @staticmethod
    def to_pandas_index(fhv: "FHValues") -> pd.Index:
        pass

    # cutoff conversion
    @staticmethod
    def cutoff_to_internal(cutoff):
        pass

    @staticmethod
    def cutoff_to_pandas(cutoff_internal):
        pass

    # frequency helper functions

    # 1. frequency extraction function,
    #    to get freq from pandas objects when needed
    # 2. frequency normalization function,
    #    to convert pandas freq strings to a canonical form
    # 3. offset handler

    # coerce function, a pandas-aware wrapper around FHValues
    # coercion that can handle pandas types as input
