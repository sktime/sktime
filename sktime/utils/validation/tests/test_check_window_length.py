#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
import pytest

from sktime.utils.validation import check_window_length


@pytest.mark.parametrize(
    "arg,expected",
    [
        (0.2, 1),
        (43, 43),
        (None, None),
    ],
)
def test_check_window_length(arg, expected):
    assert check_window_length(arg) == expected


@pytest.mark.parametrize("arg", ["string", -5, -0.5, 6.1, -2.4])
def test_window_length_bad_arg(arg):
    with pytest.raises(ValueError):
        check_window_length(arg)
