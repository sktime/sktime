#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = "Markus Löning"

# adapted from https://github.com/scikit-learn/scikit-learn/blob/master
# /sklearn/setup.py

import os

from setuptools import find_packages
from sktime._build_utils import maybe_cythonize_extensions


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('sktime', parent_package, top_path)

    for package in find_packages('sktime'):
        config.add_subpackage(package)

    maybe_cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
