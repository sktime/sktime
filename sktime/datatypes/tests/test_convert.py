# -*- coding: utf-8 -*-
"""Testing machine type converters for scitypes."""

__author__ = ["fkiraly"]

from sktime.datatypes import SCITYPE_REGISTER, scitype_to_mtype
from sktime.datatypes._convert import _conversions_defined, convert
from sktime.datatypes._examples import get_examples
from sktime.utils._testing.deep_equals import deep_equals

SCITYPES = [sci[0] for sci in SCITYPE_REGISTER]

# scitypes which have no conversions defined
# should be listed here to avoid false positive test errors
SCITYPES_NO_CONVERSIONS = ["Alignment"]


def _generate_fixture_tuples():
    """Return fixture tuples for pytest_generate_tests."""
    # collect fixture tuples here
    fixture_tuples = []

    for scitype in SCITYPES:

        # if we know there are no conversions defined, skip this scitype
        if scitype in SCITYPES_NO_CONVERSIONS:
            continue

        conv_mat = _conversions_defined(scitype)

        mtypes = scitype_to_mtype(scitype, softdeps="exclude")

        if len(mtypes) == 0:
            # if there are no mtypes, this must have been reached by mistake/bug
            raise RuntimeError("no mtypes defined for scitype " + scitype)

        # by convention, number of examples is the same for all mtypes of the scitype
        examples = get_examples(mtype=mtypes[0], as_scitype=scitype, return_lossy=True)
        n_fixtures = len(examples)

        # there must be fixtures for each scitype, otherwise there is a bug in the tests
        if n_fixtures == 0:
            raise RuntimeError("no fixtures defined for scitype " + scitype)

        for to_type in mtypes:
            for from_type in mtypes:
                for i in range(n_fixtures):
                    # only add if conversion is implemented
                    if conv_mat[to_type][from_type]:
                        fixture_tuples += [(scitype, from_type, to_type, i)]

    return fixture_tuples


def pytest_generate_tests(metafunc):
    """Test parameterization routine for pytest.

    Fixtures parameterized
    ----------------------
    scitype : str - scitypes
    from_mtype : str - mtype of "from" conversion to test, belongs to scitype
    to_mtype : str - mtype of conversion target ("to") to test, belongs to scitype
    fixture_index : int - index of fixture tuple use for conversion
    """
    # we assume all four arguments are present in the test below
    keys = _generate_fixture_tuples()

    ids = []
    for tuple in keys:
        ids += [f"{tuple[0]}-from:{tuple[1]}-to:{tuple[2]}-fixture:{tuple[3]}"]

    # parameterize test with from-mtpes
    metafunc.parametrize("scitype,from_mtype,to_mtype,fixture_index", keys, ids=ids)


def test_convert(scitype, from_mtype, to_mtype, fixture_index):
    """Tests that conversions for scitype agree with from/to example fixtures.

    Parameters
    ----------
    scitype : str - scitypes
    from_mtype : str - mtype of "from" conversion to test, belongs to scitype
    to_mtype : str - mtype of conversion target ("to") to test, belongs to scitype
    from_fixture : int - index of fixture tuple use for conversion

    Raises
    ------
    AssertionError if a converted object does not match fixture
    error if conversion itself raises an error
    """
    # retrieve from/to fixture for conversion
    from_fixture = get_examples(
        mtype=from_mtype, as_scitype=scitype, return_lossy=True
    ).get(fixture_index)

    to_fixture = get_examples(
        mtype=to_mtype, as_scitype=scitype, return_lossy=True
    ).get(fixture_index)

    # retrieve indicators whether conversion makes sense
    # to-fixture is in example dict and is not None
    cond1 = to_fixture is not None and to_fixture[0] is not None
    # from-fixture is in example dict and is not None
    cond2 = from_fixture is not None and from_fixture[0] is not None
    # from-fixture is not None and not lossy
    cond3 = cond2 and from_fixture[1] is not None and not from_fixture[1]

    msg = (
        f"conversion {from_mtype} to {to_mtype} failed for fixture {fixture_index}, "
        "expected result (y) and converted result (x) are not equal because: "
    )

    # test that converted from-fixture equals to-fixture
    if cond1 and cond2 and cond3:

        converted_fixture_i = convert(
            obj=from_fixture[0],
            from_type=from_mtype,
            to_type=to_mtype,
            as_scitype=scitype,
        )

        equals, deep_equals_msg = deep_equals(
            converted_fixture_i,
            to_fixture[0],
            return_msg=True,
        )
        assert equals, msg + deep_equals_msg
