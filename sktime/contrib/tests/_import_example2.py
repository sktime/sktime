# -*- coding: utf-8 -*-
"""Trying to resolve import issue."""

__author__ = ["TonyBagnall"]
from sktime.contrib.tests import public_function1


def public_function2():
    print("We hate tottenham")
    public_function1()


public_function2()
