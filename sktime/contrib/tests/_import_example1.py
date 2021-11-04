# -*- coding: utf-8 -*-
"""Trying to resolve import issue."""

__author__ = ["TonyBagnall"]
from sktime.contrib.tests import public_function2


def public_function1():
    print("Up the arsenal")
    public_function2()


public_function1()
