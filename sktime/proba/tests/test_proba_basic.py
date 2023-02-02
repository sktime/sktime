# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]


def test_proba_example():

    from sktime.proba.tfd import Normal

    n = Normal(mean=[[0, 1], [2, 3], [4, 5]], sd=1)

    assert n.shape == (3, 2)

    one_row = n.loc[[1]]
    assert isinstance(one_row, Normal)
    assert one_row.shape == (1, 2)
