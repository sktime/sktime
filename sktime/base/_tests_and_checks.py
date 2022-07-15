#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""collection of different verificiation and checking classes that can be used in composition."""

import abc
import warnings

__author__ = ["miraep8"]
__all__ = ["LengthChecker"]


class BaseChecker(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def check(self, **kwargs):
        """Contains the check logic."""
        pass


class LengthChecker(BaseChecker):
    """For input check that length provided are all the same."""

    def __init__(self, to_check: list = [], ignore_none: bool = False):
        self.to_check = to_check
        self.ignore_none = ignore_none

    def _get_len_of_element(self, name, kwargs):
        if not name in kwargs.keys():
            raise ValueError(
                f"Name {name} is not a valid argument for this \n"
                "estimator and thus its length cannot be checked."
            )
        if hasattr(kwargs[name], "__len__"):
            return len(kwargs[name])
        if not kwargs[name] and self.ignore_none:
            return 0
        raise ValueError(
            f"Attempting to check the length of an object {name} \n"
            'which doesn\'t have a built in "length".  Remove  \n'
            f"{name} from the list of arguments whose length should\n"
            "be compared."
        )

    def check(self, **kwargs):
        """Check that all arguments in named_test are the same type."""

        if not self.to_check:
            return

        if len(self.to_check) == 1 and hasattr(kwargs[self.to_check[0]], "__len__"):
            warnings.warn(
                "LengthChecker class was only passed one argument to \n"
                "check the length of. An object is the same length as \n"
                "itself, this isn't an issue, but we wonder - was it \n"
                "what you intended?"
            )
            return

        lens = [self._get_len_of_element(name, kwargs) for name in self.to_check]
        if not len(set(lens)) == 1:
            raise ValueError(
                f"Length Checker was run for inputs {self.to_check} \n"
                "however not all parameters were the same length. \n"
                f"Instead the lengths observed were: {lens}"
            )
