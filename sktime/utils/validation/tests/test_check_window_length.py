#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

from sktime.utils.validation.__init__ import check_window_length


def test_check_window_length(window_length):
    assert check_window_length(0.2) == 1  # true
    assert check_window_length(43) == 43  # True
    assert check_window_length(None) is None  # True
