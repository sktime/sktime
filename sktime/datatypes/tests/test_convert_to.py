"""Testing machine type converters for scitypes - convert_to utility."""

__author__ = ["fkiraly"]

import pytest

from sktime.datatypes._convert import convert_to
from sktime.datatypes._examples import get_examples
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.deep_equals import deep_equals

# hard-coded scitypes/mtypes to use in test_convert_to
#   easy to change in case the strings change
SCITYPES = ["Series", "Panel"]
MTYPES_SERIES = ["pd.Series", "np.ndarray", "pd.DataFrame"]
MTYPES_PANEL = ["pd-multiindex", "df-list", "numpy3D"]


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.datatypes", "sktime.utils.deep_equals"]),
    reason="Test only if sktime.datatypes or utils.deep_equals has been changed",
)
def test_convert_to_simple():
    """Testing convert_to basic call works."""
    scitype = SCITYPES[0]

    from_fixt = get_examples(mtype=MTYPES_SERIES[1], as_scitype=scitype).get(0)
    # expectation is that the conversion is to mtype MTYPES_SERIES[0]
    exp_fixt = get_examples(mtype=MTYPES_SERIES[0], as_scitype=scitype).get(0)

    # carry out the conversion using convert_to
    converted = convert_to(from_fixt, to_type=MTYPES_SERIES[0], as_scitype=scitype)

    # compare expected output with actual output of convert_to
    msg = "convert_to basic call does not seem to work."
    assert deep_equals(converted, exp_fixt), msg


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.datatypes", "sktime.utils.deep_equals"]),
    reason="Test only if sktime.datatypes or utils.deep_equals has been changed",
)
def test_convert_to_without_scitype():
    """Testing convert_to call without scitype specification."""
    scitype = SCITYPES[0]

    from_fixt = get_examples(mtype=MTYPES_SERIES[1], as_scitype=scitype).get(0)
    # convert_to should recognize the correct scitype, otherwise same as above
    exp_fixt = get_examples(mtype=MTYPES_SERIES[0], as_scitype=scitype).get(0)

    # carry out the conversion using convert_to
    converted = convert_to(from_fixt, to_type=MTYPES_SERIES[0])

    # compare expected output with actual output of convert_to
    msg = "convert_to call without scitype does not seem to work."
    assert deep_equals(converted, exp_fixt), msg


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.datatypes", "sktime.utils.deep_equals"]),
    reason="Test only if sktime.datatypes or utils.deep_equals has been changed",
)
def test_convert_to_mtype_list():
    """Testing convert_to call to_type being a list, of same scitype."""
    # convert_to list
    target_list = MTYPES_SERIES[:2]
    scitype = SCITYPES[0]

    # example that is on the list
    from_fixt_on = get_examples(mtype=MTYPES_SERIES[1], as_scitype=scitype).get(0)
    # example that is not on the list
    from_fixt_off = get_examples(mtype=MTYPES_SERIES[2], as_scitype=scitype).get(0)

    # if on the list, result should be equal to input
    exp_fixt_on = get_examples(mtype=MTYPES_SERIES[1], as_scitype=scitype).get(0)
    # if off the list, result should be converted to mtype that is first on the list
    exp_fixt_off = get_examples(mtype=MTYPES_SERIES[0], as_scitype=scitype).get(0)

    # carry out the conversion using convert_to
    converted_on = convert_to(from_fixt_on, to_type=target_list)
    converted_off = convert_to(from_fixt_off, to_type=target_list)

    # compare expected output with actual output of convert_to
    msg = "convert_to call does not work with list for to_type."
    assert deep_equals(converted_on, exp_fixt_on), msg
    assert deep_equals(converted_off, exp_fixt_off), msg


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.datatypes", "sktime.utils.deep_equals"]),
    reason="Test only if sktime.datatypes or utils.deep_equals has been changed",
)
def test_convert_to_mtype_list_different_scitype():
    """Testing convert_to call to_type being a list, of different scitypes."""
    # convert_to list
    target_list = MTYPES_SERIES[:2] + MTYPES_PANEL[:2]
    scitype0 = SCITYPES[0]
    scitype1 = SCITYPES[1]

    # example that is on the list and of scitype0
    from_fixt_on_0 = get_examples(mtype=MTYPES_SERIES[1], as_scitype=scitype0).get(0)
    # example that is not on the list and of scitype0
    from_fixt_off_0 = get_examples(mtype=MTYPES_SERIES[2], as_scitype=scitype0).get(0)
    # example that is on the list and of scitype1
    from_fixt_on_1 = get_examples(mtype=MTYPES_PANEL[1], as_scitype=scitype1).get(0)
    # example that is not on the list and of scitype1
    from_fixt_off_1 = get_examples(mtype=MTYPES_PANEL[2], as_scitype=scitype1).get(0)

    # if on the list, result should be equal to input
    exp_fixt_on_0 = get_examples(mtype=MTYPES_SERIES[1], as_scitype=scitype0).get(0)
    exp_fixt_on_1 = get_examples(mtype=MTYPES_PANEL[1], as_scitype=scitype1).get(0)
    # if off the list, result should be converted to mtype
    #   of the same scitype that appears earliest on the list
    exp_fixt_off_0 = get_examples(mtype=MTYPES_SERIES[0], as_scitype=scitype0).get(0)
    exp_fixt_off_1 = get_examples(mtype=MTYPES_PANEL[0], as_scitype=scitype1).get(0)

    # carry out the conversion using convert_to
    converted_on_0 = convert_to(from_fixt_on_0, to_type=target_list)
    converted_off_0 = convert_to(from_fixt_off_0, to_type=target_list)
    converted_on_1 = convert_to(from_fixt_on_1, to_type=target_list)
    converted_off_1 = convert_to(from_fixt_off_1, to_type=target_list)

    # compare expected output with actual output of convert_to
    msg = "convert_to call does not work with list for to_type of different scitypes."
    assert deep_equals(converted_on_0, exp_fixt_on_0), msg
    assert deep_equals(converted_off_0, exp_fixt_off_0), msg
    assert deep_equals(converted_on_1, exp_fixt_on_1), msg
    assert deep_equals(converted_off_1, exp_fixt_off_1), msg
