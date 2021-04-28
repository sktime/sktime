#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
import pytest

from sktime.utils.validation import check_window_length


def test_check_window_length():
    # assert check_window_length(0.2) == 1  # true
    # assert check_window_length(43) == 43  # True
    assert check_window_length(None) is None  # True


def test_raises_exception_on_string_arg():
    with pytest.raises(TypeError):
        check_window_length("Any String")


def test_negative_integer_window_length():
    with pytest.raises(ValueError):
        check_window_length(-1)


def test_negative_float_window_length():
    with pytest.raises(ValueError):
        check_window_length(-0.1)


def test_large_float_window_length():
    with pytest.raises(ValueError):
        check_window_length(6.1)
