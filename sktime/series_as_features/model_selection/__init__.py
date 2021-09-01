#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = [
    "SingleSplit",
    "PresplitFilesCV",
]

from sktime.series_as_features.model_selection._split import PresplitFilesCV
from sktime.series_as_features.model_selection._split import SingleSplit
