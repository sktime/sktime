#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement transformers for summarizing a time series."""

import warnings

from sktime.transformations.series.summarize import WindowSummarizer

__all__ = ["WindowSummarizer"]

warnings.warn(
    "WindowSummarizer has been moved to transformations.series.summarize,"
    + " the old location in series.windows_summarize is deprecated since 0.11.0,"
    + " and will be removed in 0.12.0. Please use the import from "
    + "transformations.series.summarize import WindowSummarizer."
)
