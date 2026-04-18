#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapters for time series detection."""

__all__ = ["PyODAnnotator", "PyODDetector"]

try:
    from sktime.detection.adapters._pyod import PyODAnnotator, PyODDetector
except Exception:
    # Optional dependency not available (e.g., during docs build)
    PyODAnnotator = None
    PyODDetector = None