# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
ForecastingHorizonV2: pandas-agnostic forecasting horizon implementation.

All pandas-specific logic (type conversions, frequency handling, version detection)
is delegated to the _fh_utils module.
"""
