"""Testing mtype/scitypes lookup."""

__author__ = ["fkiraly"]

from sktime.datatypes._registry import (
    MTYPE_SOFT_DEPS,
    generate_mtype_register,
    mtype_to_scitype,
    scitype_to_mtype,
)
from sktime.utils.dependencies import _check_soft_dependencies


def pytest_generate_tests(metafunc):
    register = generate_mtype_register()
    if "mtype" in metafunc.fixturenames and "scitype" in metafunc.fixturenames:
        mtypes_scitypes = [(k[0], k[1]) for k in register]
        metafunc.parametrize("mtype, scitype", mtypes_scitypes)
    elif "mtype" in metafunc.fixturenames:
        mtypes = [k[0] for k in register]
        metafunc.parametrize("mtype", mtypes)
    elif "scitype" in metafunc.fixturenames:
        scitypes = [k[1] for k in register]
        metafunc.parametrize("scitype", scitypes)


def test_mtype_to_scitype(mtype, scitype):
    """Tests that mtype_to_scitype yields the correct output for a string.

    Parameters
    ----------
    mtype : str - mtype string, mtype belonging to scitype
    scitype : str - scitype string, scitype of mtype

    Raises
    ------
    AssertionError mtype_to_scitype does not convert mtype to scitype
    Exception if any is raised by mtype_to_scitype
    """
    result = mtype_to_scitype(mtype)
    msg = (
        f'mtype_to_scitype does not correctly convert mtype "{mtype}" to scitype '
        f'"{scitype}", returned {result}'
    )
    assert result == scitype, msg


def test_mtype_to_scitype_list():
    """Tests that mtype_to_scitype yields the correct output for a list.

    Parameters
    ----------
    mtype : str - mtype string
    scitype : str - scitype string, belonging to mtype

    Raises
    ------
    AssertionError mtype_to_scitype does not convert mtype to scitype
    Exception if any is raised by mtype_to_scitype
    """
    MTYPE_REGISTER = generate_mtype_register()
    mtype_list = [k[0] for k in MTYPE_REGISTER]
    expected_scitype_list = [k[1] for k in MTYPE_REGISTER]
    result = mtype_to_scitype(mtype_list)
    msg = (
        "mtype_to_scitype does not correctly convert list of mtypes to list of scitypes"
    )
    assert result == expected_scitype_list, msg


def test_scitype_to_mtype(mtype, scitype):
    """Tests that scitype_to_mtype yields the correct output for a string.

    Parameters
    ----------
    mtype : str - mtype string, mtype belonging to scitype
    scitype : str - scitype string, scitype of mtype

    Raises
    ------
    AssertionError scitype_to_mtype does not return correct list of mtypes
    Exception if any is raised by scitype_to_mtype
    """
    # check that mtype is always returned in "all" setting
    result = scitype_to_mtype(scitype, softdeps="all")
    msg = (
        f"scitype_to_mtype does not correctly retrieve all mtypes for scitype "
        f'"{scitype}", mtype "{mtype}" is missing from result returned: {result}'
    )
    assert mtype in result, msg

    # check that mtype is not returned in "exclude" setting if requires soft dep
    result_no_softdeps = scitype_to_mtype(scitype, softdeps="exclude")
    assert (mtype in MTYPE_SOFT_DEPS.keys()) != (mtype in result_no_softdeps)

    if mtype in MTYPE_SOFT_DEPS.keys():
        softdep_present = _check_soft_dependencies(
            MTYPE_SOFT_DEPS[mtype], severity="none"
        )
    else:
        softdep_present = True

    # check that mtype is returned for "present" setting iff soft dep is satisfied
    result_present = scitype_to_mtype(scitype, softdeps="present")
    assert (mtype in result_present) == softdep_present
