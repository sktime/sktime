# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements wrappers for estimators from hmmlearn."""

__all__ = [
    "BaseHMMLearn",
    "GaussianHMM",
    "GMMHMM",
]

from sktime.annotation.hmm_learn.base import BaseHMMLearn
from sktime.annotation.hmm_learn.gaussian import GaussianHMM
from sktime.annotation.hmm_learn.gmm import GMMHMM
