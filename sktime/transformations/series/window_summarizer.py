#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement transformers for summarizing a time series."""

import warnings

from sktime.transformations.series.summarize import WindowSummarizer

warnings.warn(
    "WindowSummarizer will be moved to summarize. In future, use"
    + "sktime.transformations.series.summarize import WindowSummarizer"
)

WindowSummarizer()
