# -*- coding: utf-8 -*-
"""Testing machine type converters for scitypes."""

__author__ = ["fkiraly"]

import pytest

from sktime.datatypes._convert import convert, _conversions_defined
from sktime.datatypes._examples import get_examples
from sktime.datatypes import SCITYPE_REGISTER, Scitype, PanelMtype, SeriesMtype
from sktime.datatypes._convert import _find_conversion_path

from sktime.utils._testing.deep_equals import deep_equals

SCITYPES = [sci[0] for sci in SCITYPE_REGISTER] + [scitype for scitype in Scitype]

ENUM_SERIES_MTYPES = [mtype for mtype in SeriesMtype]
ENUM_PANEL_MTYPES = [mtype for mtype in PanelMtype]


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

    if str(scitype) == "Series":
        mtypes = [*mtypes, *ENUM_SERIES_MTYPES]
    else:
        mtypes = [*mtypes, *ENUM_PANEL_MTYPES]

    if len(mtypes) == 0:
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
                to_fixture = fixtures[str(to_type)].get(i)
                from_fixture = fixtures[str(from_type)].get(i)

                # retrieve indicators whether conversion makes sense
                # to-fixture is in example dict and is not None
                cond1 = to_fixture is not None and to_fixture[0] is not None
                # from-fixture is in example dict and is not None
                cond2 = from_fixture is not None and from_fixture[0] is not None
                # from-fixture is not None and not lossy
                cond3 = cond2 and from_fixture[1] is not None and not from_fixture[1]
                # conversion is implemented
                cond4 = conv_mat[str(to_type)][str(from_type)]

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
                elif cond1 and cond2:
                    # Test trying to find an indirect path
                    try:
                        # Try used as path maybe not found
                        indirect_path = convert(
                            obj=from_fixture[0],
                            from_type=from_type,
                            to_type=to_type,
                            as_scitype=scitype,
                        )
                        assert deep_equals(
                            indirect_path,
                            to_fixture[0],
                        ), msg

                    except Exception:
                        continue


@pytest.mark.parametrize("scitype", SCITYPES)
def test_find_conversion_path(scitype):
    """Tests conversion path.

    Parameters
    ----------
    scitype: str
        Current scitypes to check path over

    Raises
    ------
    AssertionError if the conversion path fails
    """
    if scitype == "Panel":
        resolve_enum = PanelMtype
    else:
        resolve_enum = SeriesMtype
    conv_mat = _conversions_defined(scitype)
    mtypes = conv_mat.index.values
    for from_type in mtypes:
        for to_type in mtypes:
            path_lossy = _find_conversion_path(from_type, to_type)
            path_no_lossy = _find_conversion_path(from_type, to_type, allow_lossy=False)
            if len(path_lossy) > 0:
                assert path_lossy[-1] is to_type
            if len(path_no_lossy) > 0:
                if resolve_enum[from_type] is False:
                    for val in path_no_lossy:
                        assert resolve_enum[val].is_lossy is False
                assert path_no_lossy[-1] is to_type
