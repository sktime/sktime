__author__ = ["Markus Löning"]
__all__ = ["test_validate_fh_bad_input_args"]

import numpy as np
import pytest
from pytest import raises

from sktime.utils.validation.forecasting import validate_fh


bad_input_args = (
    (1, 2),  # tuple
    [],  # empty list
    np.array([]),  # empty array
    'some_string',  # string but not "insample"
    0.1,  # float
    [0.1, 2],  # float in list
    np.array([0.1, 2]),  # float in list
    True,  # boolean
    [True, 2],  # boolean in list
    np.array([False, 2]),  # boolean in array
)


@pytest.mark.parametrize("arg", bad_input_args)
def test_validate_fh_bad_input_args(arg):
    with raises(ValueError):
        validate_fh(arg)
