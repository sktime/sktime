# -*- coding: utf-8 -*-
"""Testing machine type checkers for scitypes."""

__author__ = ["fkiraly"]

import numpy as np
import pytest

from sktime.datatypes import MTYPE_REGISTER, SCITYPE_REGISTER
from sktime.datatypes._check import check_dict, check_is_mtype
from sktime.datatypes._check import mtype as infer_mtype
from sktime.datatypes._examples import get_examples

SCITYPES = [sci[0] for sci in SCITYPE_REGISTER]

# scitypes where mtype inference is not unique
# alignment is excluded since mtypes can be ambiguous
#   (indices could be both loc or iloc when integers)
SCITYPES_AMBIGUOUS_MTYPE = ["Alignment"]


@pytest.mark.parametrize("scitype", SCITYPES)
def test_check_positive(scitype):
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
    if scitype not in [s[0] for s in SCITYPE_REGISTER]:
        raise RuntimeError(scitype + " is not in the SCITYPE_REGISTER")
    mtypes = [key[0] for key in MTYPE_REGISTER if key[1] == scitype]

    if len(mtypes) == 0:
        raise RuntimeError("no mtypes defined for scitype " + scitype)

    fixtures = dict()

    for mtype in mtypes:
        # if we don't do this we get into a clash between linters
        mtype_long_variable_name_to_avoid_linter_clash = mtype
        fixtures[mtype] = get_examples(
            mtype=mtype_long_variable_name_to_avoid_linter_clash,
            as_scitype=scitype,
        )

    n_fixtures = np.max([len(fixtures[mtype]) for mtype in mtypes])

    if n_fixtures == 0:
        raise RuntimeError("no fixtures defined for scitype " + scitype)

    for i in range(n_fixtures):
        for mtype in mtypes:
            # retrieve fixture for checking
            fixture = fixtures[mtype].get(i)

            # todo: possibly remove this once all checks are defined
            check_is_defined = (mtype, scitype) in check_dict.keys()

            # check fixtures that exist against checks that exist
            if fixture is not None and check_is_defined:
                check_result = check_is_mtype(
                    fixture, mtype, scitype, return_metadata=True
                )
                if not check_result[0]:
                    msg = (
                        f"check_is_mtype returns False on {mtype} "
                        f"fixture {i}, message: "
                    )
                    msg = msg + check_result[1]
                assert check_result[0], msg


@pytest.mark.parametrize("scitype", SCITYPES)
def test_check_negative(scitype):
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
    if scitype not in [s[0] for s in SCITYPE_REGISTER]:
        raise RuntimeError(scitype + " is not in the SCITYPE_REGISTER")
    if scitype in SCITYPES_AMBIGUOUS_MTYPE:
        return None
    mtypes = [key[0] for key in MTYPE_REGISTER if key[1] == scitype]

    if len(mtypes) == 0:
        raise RuntimeError("no mtypes defined for scitype " + scitype)

    fixtures = dict()

    for mtype in mtypes:
        # if we don't do this we get into a clash between linters
        mtype_long_variable_name_to_avoid_linter_clash = mtype
        fixtures[mtype] = get_examples(
            mtype=mtype_long_variable_name_to_avoid_linter_clash,
            as_scitype=scitype,
        )

    n_fixtures = np.max([len(fixtures[mtype]) for mtype in mtypes])

    if n_fixtures == 0:
        raise RuntimeError("no fixtures defined for scitype " + scitype)

    for i in range(n_fixtures):
        for mtype in mtypes:
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


@pytest.mark.parametrize("scitype", SCITYPES)
def test_mtype_infer(scitype):
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
    if scitype not in [s[0] for s in SCITYPE_REGISTER]:
        raise RuntimeError(scitype + " is not in the SCITYPE_REGISTER")
    if scitype in SCITYPES_AMBIGUOUS_MTYPE:
        return None
    mtypes = [key[0] for key in MTYPE_REGISTER if key[1] == scitype]

    if len(mtypes) == 0:
        raise RuntimeError("no mtypes defined for scitype " + scitype)

    fixtures = dict()

    for mtype in mtypes:
        # if we don't do this we get into a clash between linters
        mtype_long_variable_name_to_avoid_linter_clash = mtype
        fixtures[mtype] = get_examples(
            mtype=mtype_long_variable_name_to_avoid_linter_clash,
            as_scitype=scitype,
        )

    n_fixtures = np.max([len(fixtures[mtype]) for mtype in mtypes])

    if n_fixtures == 0:
        raise RuntimeError("no fixtures defined for scitype " + scitype)

    for i in range(n_fixtures):
        for mtype in mtypes:
            # retrieve fixture for checking
            fixture = fixtures[mtype].get(i)

            # todo: possibly remove this once all checks are defined
            check_is_defined = (mtype, scitype) in check_dict.keys()

            # check fixtures that exist against checks that exist
            if fixture is not None and check_is_defined:
                assert mtype == infer_mtype(
                    fixture, as_scitype=scitype
                ), f"mtype {mtype} not correctly identified for fixture {i}"
