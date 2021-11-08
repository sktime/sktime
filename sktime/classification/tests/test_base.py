# -*- coding: utf-8 -*-
"""Unit tests for classifier base class functionality. """

__author__ = ["mloning", "fkiraly", "TonyBagnall", "MatthewMiddlehurst"]

import pytest

from sktime.classification.base import BaseClassifier, check_capabilities

def test_check_capabilities():
    """Test the reading of data characteristics. There are eight different
    combinations to be tested with a classifier that can handle it and that cannot.
    """
    handles_none = BaseClassifier()
    handles_all = BaseClassifier()
    handles_all._tags["capability:multivariate"] = True
    handles_all._tags["capability:unequal_length"] = True
    handles_all._tags["capability:missing_values"] = True

    pytest.raises(check_capabilities(handles_none,True, True, True))
    # pytest.raises(check_capabilities(handles_none, True, True, False))
    # pytest.raises(check_capabilities(handles_none, True, False, True))
    # pytest.raises(check_capabilities(handles_none, False, True, True))
    # pytest.raises(check_capabilities(handles_none, True, False, False))
    # pytest.raises(check_capabilities(handles_none, False, True, False))
    # pytest.raises(check_capabilities(handles_none, False, False, True))
    # check_capabilities(handles_none, False, False, False)
    # check_capabilities(handles_none, True, True, True)
    # check_capabilities(handles_none, True, True, False)
    # check_capabilities(handles_none, True, False, True)
    # check_capabilities(handles_none, False, True, True)
    # check_capabilities(handles_none, True, False, False)
    # check_capabilities(handles_none, False, True, False)
    # check_capabilities(handles_none, False, False, True)
    # check_capabilities(handles_none, False, False, False)


