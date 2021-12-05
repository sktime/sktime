# -*- coding: utf-8 -*-
"""Testing machine type converters for scitypes."""

__author__ = ["fkiraly"]

import pytest

from sktime.datatypes import SCITYPE_REGISTER
from sktime.datatypes._convert import _conversions_defined, convert
from sktime.datatypes._examples import get_examples
from sktime.utils._testing.deep_equals import deep_equals

SCITYPES = [sci[0] for sci in SCITYPE_REGISTER]

# scitypes which have no conversions defined
# should be listed here to avoid false positive test errors
SCITYPES_NO_CONVERSIONS = ["Alignment"]


@pytest.mark.parametrize("scitype", SCITYPES)
def test_convert(scitype):
    """Tests that conversions for scitype agree with from/to example fixtures.

    Parameters
    ----------
    scitype : str - name of scitype for which mtype conversions are tested

    Raises
    ------
    AssertionError if a converted object does not match fixture
    error if conversion itself raises an error
    """
    conv_mat = _conversions_defined(scitype)
    mtypes = conv_mat.index.values

    if len(mtypes) == 0:
        # if we know there are no conversions defined, skip this test
        # otherwise this must have been reached by mistake/bug
        if scitype in SCITYPES_NO_CONVERSIONS:
            return None
        else:
            raise RuntimeError("no mtypes defined for scitype " + scitype)

    fixtures = dict()

    for mtype in mtypes:
        # if we don't do this we get into a clash between linters
        mtype_long_variable_name_to_avoid_linter_clash = mtype
        fixtures[mtype] = get_examples(
            mtype=mtype_long_variable_name_to_avoid_linter_clash,
            as_scitype=scitype,
            return_lossy=True,
        )

    if len(fixtures[mtypes[0]]) == 0:
        raise RuntimeError("no fixtures defined for scitype " + scitype)

    # by convention, all fixtures are mirrored across all mtypes
    #  so len(fixtures[mtypes[i]]) does not depend on i
    n_fixtures = len(fixtures[mtypes[0]])

    for i in range(n_fixtures):
        for from_type in mtypes:
            for to_type in mtypes:

                # retrieve from/to fixture for conversion
                to_fixture = fixtures[to_type].get(i)
                from_fixture = fixtures[from_type].get(i)

                # retrieve indicators whether conversion makes sense
                # to-fixture is in example dict and is not None
                cond1 = to_fixture is not None and to_fixture[0] is not None
                # from-fixture is in example dict and is not None
                cond2 = from_fixture is not None and from_fixture[0] is not None
                # from-fixture is not None and not lossy
                cond3 = cond2 and from_fixture[1] is not None and not from_fixture[1]
                # conversion is implemented
                cond4 = conv_mat[to_type][from_type]

                msg = f"conversion {from_type} to {to_type} failed for fixture {i}"

                # test that converted from-fixture equals to-fixture
                if cond1 and cond2 and cond3 and cond4:

                    converted_fixture_i = convert(
                        obj=from_fixture[0],
                        from_type=from_type,
                        to_type=to_type,
                        as_scitype=scitype,
                    )

                    assert deep_equals(
                        converted_fixture_i,
                        to_fixture[0],
                    ), msg
