#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""A minimal BaseObject and BaseEstimator."""

__author__ = ["miraep8"]
__all__ = ["SimpleBaseObject", "SimpleBaseEstimator"]

import abc


class SimpleBaseObject:
    """Simple class that facilitates getting and setting params for all estimators"""

    input_checks = []

    def run_input_checks(self, kwargs):
        for check in self.input_checks:
            check.check(kwargs)

    def get_params(self, name_str="", deep=True):
        """Get all parameters for this Estimator."""
        param_return = {}
        for parameter in self.parameters.keys():
            param_key = "".join([name_str, parameter])
            param_return[param_key] = getattr(self, parameter)
            if deep and isinstance(param_return[param_key], SimpleBaseEstimator):
                param_return.update(
                    param_return[param_key].get_params(
                        name_str="".join([parameter, "__"])
                    )
                )
        return param_return

    def make_list(self, obj):
        """Make obj a list if it isn't already."""
        if isinstance(obj, list):
            return obj
        return [obj]

    def set_params(self, **kwargs):
        """Update all parameters for valid parameters and update types."""
        for parameter, value in kwargs.items():
            if parameter in self.parameters.keys() and type(value) in self.make_list(
                self.parameters[parameter]
            ):
                setattr(self, parameter, value)
            else:
                raise ValueError(
                    "".join(
                        [
                            f"{type(self).__name__} doesn't have a parameter ",
                            f"{parameter} which accepts input of type",
                            f"{type(value).__name__}. ",
                            f"Valid options and types are {self.parameters}",
                        ]
                    )
                )


class SimpleBaseEstimator(SimpleBaseObject, metaclass=abc.ABCMeta):
    """An experiment in a more simple Base Estimator."""

    @abc.abstractmethod
    def fit(self):
        """Fit."""
        pass

    @abc.abstractmethod
    def predict(self):
        """Predict."""
        pass
