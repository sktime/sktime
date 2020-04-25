#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["_construct_instance"]

from sktime.utils.testing.construct.config import TEST_CONSTRUCT_CONFIG_LOOKUP


def _construct_instance(Estimator):
    """Construct Estimator instance if possible"""

    # some estimators require parameters during construction
    required_parameters = getattr(Estimator, "_required_parameters", [])

    # construct with parameters
    if len(required_parameters) > 0:
        # look up default instantiations for estimators which require
        # arguments during constructions
        parameters = {}
        if Estimator in TEST_CONSTRUCT_CONFIG_LOOKUP:
            parameters = TEST_CONSTRUCT_CONFIG_LOOKUP[Estimator]
        if len(parameters) == 0:
            raise ValueError(f"Estimator: {Estimator} requires parameters for construction, "
                             f"but no test configuration has been found.")
        estimator = Estimator(**parameters)

    # construct without parameters if none are required
    else:
        estimator = Estimator()

    return estimator
