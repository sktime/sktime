# -*- coding: utf-8 -*-
"""Test.
"""

__author__ = ["TonyBagnall"]

from classification.import_bug_test import public_function1, public_function2


def import_CI_bug():
    return public_function1()+public_function1()


def import_CI_bug2():
    return public_function2()+public_function2()
