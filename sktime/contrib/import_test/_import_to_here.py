# -*- coding: utf-8 -*-
"""Test.
"""

__author__ = ["TonyBagnall"]

from sktime.contrib.import_bug_test import public_function1, public_function2


def import_CI_bug():
    return public_function1()+public_function1()


print(public_function1())