# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
ForecastingHorizonV2: pandas-agnostic forecasting horizon implementation.

All pandas-specific logic (type conversions, frequency handling, version detection)
is delegated to the _fh_utils module.
"""

__all__ = ["ForecastingHorizonV2"]

import numpy as np


class ForecastingHorizonSequence:
    # this is the new internal representation to store forecasting horizon
    def __init__():
        # must storemetadata
        pass

    @property
    def values(self) -> np.ndarray:
        pass

    @property
    def metadata(self) -> dict:
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, key):
        pass

    def __repr__(self) -> str:
        pass

    def copy(self) -> "ForecastingHorizonSequence":
        pass


class ForecastingHorizonV2:
    """Forecasting Horizon with clean, pandas-agnostic design."""

    def __init__():
        pass

    @property
    def is_relative(self) -> bool:
        pass

    @property
    def freq(self):
        pass

    def to_pandas(self):
        # Import here to defer pandas dependency
        # from __ import _fh_utils
        # return _fh_utils.convert_to_pandas(self._sequence)
        pass

    def to_numpy(self, **kwargs) -> np.ndarray:
        # return values of fh
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, key):
        pass

    def __repr__(self) -> str:
        pass

    # conversion methods to support
    # relative to absolute
    # and vice-versa

    def to_relative(self, cutoff=None) -> "ForecastingHorizonV2":
        pass

    def to_absolute(self, cutoff) -> "ForecastingHorizonV2":
        pass

    # some other methods to support requirements of forecasting horizon mentioned
    # in github issue

    def _is_contiguous(self) -> bool:
        pass

    def get_expected_pred_idx():
        """Construct DataFrame Index expected in y_pred (return of _predict)."""
        pass
        # multi-index support will require complex logic here
        # current versions is supports only RangeIndex
