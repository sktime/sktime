#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Tests for hierarchical reconcilers."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# __author__ = [""]
# __all__ = []

# import pytest

# from sktime.transformations.hierarchical import Reconciler
# from sktime.utils._testing.hierarchical import bottom_hier_datagen

# # marks for pytest to run with sub in param names with the list
# @pytest.mark.parametrize("method", ["ols", "bu"])
# def test_reconciler_fit_transform(param_names):
#     """Tests fit_trasnform and output of reconciler.

#     Raises
#     ------
#     Exceopion if Renconciler.fit_transform does not run
#     AssertionError if output has wrong format (not reconciled)
#     """
#     reconciler = Reconciler(method=method)
#     X = bottom_hier_datagen()
#     X = aggregate_stuff(X)

#     Xt = reconciler.fit_transform(X)
#     asswert is_reconciled(Xt), "Reconciler does not return reconciled Xt"
