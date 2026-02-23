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

import numpy as np
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
    def cutoff_to_internal(cutoff, freq=None):
        """Convert cutoff to internal representation.

        Parameters
        ----------
        cutoff : pd.Period, pd.Timestamp, int, pd.Index, or np.integer
            Cutoff value. If pd.Index, the last element is used.
        freq : str or None
            Frequency hint.

        Returns
        -------
        tuple of (np.int64, FHValueType, str or None, str or None)
            (value, value_type, freq, timezone)
        """
        pass

    @staticmethod
    def cutoff_to_pandas(cutoff_internal):
        pass

    @staticmethod
    def steps_to_nanos(steps: np.ndarray, freq: str, ref_nanos=None) -> np.ndarray:
        """Convert integer steps to int64 nanosecond offsets.

        Parameters
        ----------
        steps : np.ndarray of int64
            Integer step counts.
        freq : str
            Frequency string (e.g. "D", "h", "M").
        ref_nanos : int or np.int64 or None, default=None
            Reference point as nanoseconds since Unix epoch.
            Used to correctly compute offsets for variable-length periods
            (months, years). If None, uses 2000-01-01 as reference.

        Returns
        -------
        np.ndarray of int64
            Nanosecond offsets corresponding to each step.
        """
        pass

    # frequency helper functions

    # 1. frequency extraction function,
    #    to get freq from pandas objects when needed
    # 2. frequency normalization function,
    #    to convert pandas freq strings to a canonical form
    # 3. offset handler

    # coerce function, a pandas-aware wrapper around FHValues
    # coercion that can handle pandas types as input
