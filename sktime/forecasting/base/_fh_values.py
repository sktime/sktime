# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Numpy-backed internal representation for ForecastingHorizonV2.

This module is pandas-free.
All data is stored in numpy arrays with associated metadata.
This is the "core" layer that the conversion layer
feeds into and that ForecastingHorizonV2 operates on.
"""
