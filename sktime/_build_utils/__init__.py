# -*- coding: utf-8 -*-
"""Utilities useful during the build."""

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master
# /sklearn/_build_utils/__init__.py
# author: Andy Mueller, Gael Varoquaux
# license: BSD


import contextlib
import os
from distutils.version import LooseVersion

from .openmp_helpers import check_openmp_support

DEFAULT_ROOT = "sktime"
