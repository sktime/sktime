# -*- coding: utf-8 -*-
"""Testing machine type checkers for scitypes."""

__author__ = ["fkiraly"]

import numpy as np

from sktime.datatypes import MTYPE_REGISTER, SCITYPE_REGISTER
from sktime.datatypes._check import check_dict, check_is_mtype
from sktime.datatypes._check import mtype as infer_mtype
from sktime.datatypes._examples import get_examples

SCITYPES = [sci[0] for sci in SCITYPE_REGISTER]

# scitypes where mtype inference is not unique
# alignment is excluded since mtypes can be ambiguous
#   (indices could be both loc or iloc when integers)
SCITYPES_AMBIGUOUS_MTYPE = ["Alignment"]


def _get_all_mtypes_for_scitype(scitype):
    """Return list of all mtypes for scitype.

    Parameters
    ----------
    scitype : str - scitype

    Returns
    -------
    mtypes : list of str - list of mtypes for scitype
    """
    if scitype not in [s[0] for s in SCITYPE_REGISTER]:
        raise RuntimeError(scitype + " is not in the SCITYPE_REGISTER")
    mtypes = [key[0] for key in MTYPE_REGISTER if key[1] == scitype]

    if len(mtypes) == 0:
        # if there are no mtypes, this must have been reached by mistake/bug
        raise RuntimeError("no mtypes defined for scitype " + scitype)

    return mtypes


def _generate_scitype_mtype_combinations():
    """Return scitype/mtype tuples for pytest_generate_tests.

    Fixtures parameterized
    ----------------------
    scitype : str - scitype of fixture
    mtype : str - mtype of fixture
    """
    # collect fixture tuples here

    sci_mtype_tuples = []

    for scitype in SCITYPES:

        mtypes = _get_all_mtypes_for_scitype(scitype)

        for mtype in mtypes:
            sci_mtype_tuples += [(scitype, mtype)]

    return sci_mtype_tuples


def _generate_scitype_mtype_fixtureindex_combinations():
    """Return fixture tuples for pytest_generate_tests.

    Fixtures parameterized
    ----------------------
    scitype : str - scitype of fixture
    mtype : str - mtype of fixture
    fixture_index : int - index of fixture tuple with that scitype and mtype
    """
    # collect fixture tuples here

    sci_mtype_tuples = _generate_scitype_mtype_combinations()

    sci_mtype_index_tuples = []

    for tuple_j in sci_mtype_tuples:
        scitype = tuple_j[0]
        mtype = tuple_j[1]
        n_fixtures = len(get_examples(mtype=mtype, as_scitype=scitype))

        for i in range(n_fixtures):
            sci_mtype_index_tuples += [(scitype, mtype, i)]

    return sci_mtype_index_tuples


def pytest_generate_tests(metafunc):
    """Test parameterization routine for pytest.

    Fixtures parameterized
    ----------------------
    scitype : str - scitype of fixture
    mtype : str - mtype of fixture
    fixture_index : int - index of fixture tuple with that scitype and mtype
    """
    # we assume all four arguments are present in the test below

    fixturenames = set(metafunc.fixturenames)

    if set(["scitype", "mtype", "fixture_index"]).issubset(fixturenames):
        keys = _generate_scitype_mtype_fixtureindex_combinations()

        ids = []
        for tuple in keys:
            ids += [f"{tuple[0]}-{tuple[1]}-fixture:{tuple[2]}"]

        # parameterize test with from-mtpes
        metafunc.parametrize("scitype,mtype,fixture_index", keys, ids=ids)

    elif set(["scitype", "mtype"]).issubset(fixturenames):
        keys = _generate_scitype_mtype_combinations()

        ids = []
        for tuple in keys:
            ids += [f"{tuple[0]}-{tuple[1]}"]

        # parameterize test with from-mtpes
        metafunc.parametrize("scitype,mtype", keys, ids=ids)


def test_check_positive(scitype, mtype, fixture_index):
    """Tests that check_is_mtype correctly confirms the mtype of examples.

    Parameters
    ----------
    scitype : str - name of scitype for which mtype conversions are tested

    Raises
    ------
    RuntimeError if scitype is not defined or has no mtypes or examples
    AssertionError if examples are not correctly identified
    error if check itself raises an error
    """
    # retrieve fixture for checking
    fixture = get_examples(mtype=mtype, as_scitype=scitype).get(fixture_index)

    # todo: possibly remove this once all checks are defined
    check_is_defined = (mtype, scitype) in check_dict.keys()

    # check fixtures that exist against checks that exist
    if fixture is not None and check_is_defined:
        check_result = check_is_mtype(fixture, mtype, scitype, return_metadata=True)
        if not check_result[0]:
            msg = (
                f"check_is_mtype returns False on scitype {scitype}, mtype {mtype} "
                f"fixture {fixture_index}, message: "
            )
            msg = msg + check_result[1]
        assert check_result[0], msg


def test_check_metadata_inference(scitype, mtype, fixture_index):
    """Tests that check_is_mtype correctly infers metadata of examples.

    Parameters
    ----------
    scitype : str - name of scitype for which mtype conversions are tested

    Raises
    ------
    RuntimeError if scitype is not defined or has no mtypes or examples
    AssertionError if example metadata is not correctly inferred
    error if check itself raises an error
    """
    # retrieve fixture for checking
    fixture, _, expected_metadata = get_examples(
        mtype=mtype, as_scitype=scitype, return_metadata=True
    ).get(fixture_index)

    # todo: possibly remove this once all checks are defined
    check_is_defined = (mtype, scitype) in check_dict.keys()
    # if the examples have no metadata to them, don't test
    metadata_provided = expected_metadata is not None

    # check fixtures that exist against checks that exist
    if fixture is not None and check_is_defined and metadata_provided:
        check_result = check_is_mtype(fixture, mtype, scitype, return_metadata=True)
        metadata = check_result[2]

        # remove mtype & scitype key if exists, since comparison is on scitype level
        if "mtype" in metadata:
            del metadata["mtype"]
        if "scitype" in metadata:
            del metadata["scitype"]

        msg = (
            f"check_is_mtype returns wrong metadata on scitype {scitype}, "
            f"mtype {mtype}, fixture {fixture_index}. "
            f"returned: {metadata}; expected: {expected_metadata}"
        )

        assert metadata == expected_metadata, msg


def test_check_negative(scitype, mtype):
    """Tests that check_is_mtype correctly identifies wrong mtypes of examples.

    Parameters
    ----------
    scitype : str - name of scitype for which mtype conversions are tested

    Raises
    ------
    RuntimeError if scitype is not defined or has no mtypes or examples
    AssertionError if a examples are correctly identified as incompatible
    error if check itself raises an error
    """
    # if the scitype is ambiguous, we can't assume that other mtypes are negative
    if scitype in SCITYPES_AMBIGUOUS_MTYPE:
        return None

    mtypes = _get_all_mtypes_for_scitype(scitype)
    fixtures = dict()

    for other_mtype in mtypes:
        fixtures[other_mtype] = get_examples(mtype=other_mtype, as_scitype=scitype)

    n_fixtures = np.max([len(fixtures[mtype]) for mtype in mtypes])

    if n_fixtures == 0:
        raise RuntimeError("no fixtures defined for scitype " + scitype)

    for i in range(n_fixtures):
        # if mtype is not ambiguous, other mtypes are negative examples
        for wrong_mtype in list(set(mtypes).difference(set([mtype]))):

            # retrieve fixture for checking
            fixture_wrong_type = fixtures[wrong_mtype].get(i)

            # todo: possibly remove this once all checks are defined
            check_is_defined = (mtype, scitype) in check_dict.keys()

            # check fixtures that exist against checks that exist
            if fixture_wrong_type is not None and check_is_defined:
                assert not check_is_mtype(fixture_wrong_type, mtype, scitype), (
                    f"check_is_mtype {mtype} returns True "
                    f"on {wrong_mtype} fixture {i}"
                )


def test_mtype_infer(scitype, mtype, fixture_index):
    """Tests that mtype correctly infers the mtype of examples.

    Parameters
    ----------
    scitype : str - name of scitype for which mtype conversions are tested

    Raises
    ------
    RuntimeError if scitype is not defined or has no mtypes or examples
    AssertionError if mtype of examples is not correctly identified
    error if check itself raises an error
    """
    # if mtypes are ambiguous, then this test should be skipped
    if scitype in SCITYPES_AMBIGUOUS_MTYPE:
        return None

    # retrieve fixture for checking
    fixture = get_examples(mtype=mtype, as_scitype=scitype).get(fixture_index)

    # todo: possibly remove this once all checks are defined
    check_is_defined = (mtype, scitype) in check_dict.keys()

    # check fixtures that exist against checks that exist
    if fixture is not None and check_is_defined:
        assert mtype == infer_mtype(
            fixture, as_scitype=scitype
        ), f"mtype {mtype} not correctly identified for fixture {fixture_index}"
