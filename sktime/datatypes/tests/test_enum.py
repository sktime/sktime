# -*- coding: utf-8 -*-
"""File containing tests for the enum types."""
import numpy as np
import pytest
from sktime.datatypes.types import Mtype, SciType
from sktime.datatypes import (
    check_is,
    check_raise,
    convert,
    convert_to,
    mtype,
    get_examples,
    mtype_to_scitype,
    SeriesMtype,
    PanelMtype,
    Scitype,
)

ENUM_SERIES_MTYPES = [mtype for mtype in SeriesMtype]
ENUM_PANEL_MTYPES = [mtype for mtype in PanelMtype]


def _run_test(mtype_in: Mtype, scitype_in: SciType) -> None:
    """Run test for each mtype enum.

    Parameters
    ----------
    mtype_in: PanelMtype or SeriesMtype
        Enum mtype to run test on
    scitype_in: Scitype
        Scitype enum to pair with mtype to test
    """
    example = get_examples(mtype_in, scitype_in)

    if len(example) > 0:
        example = example[0]

        # test check_is
        check = check_is(example, mtype_in, scitype_in)
        assert isinstance(check, bool)
        assert check_is(example, mtype_in) is check

        # test check_raise
        if check is True:
            assert check_raise(example, mtype_in, scitype_in) is check
            assert check_raise(example, mtype_in) is check
        else:
            with pytest.raises(TypeError):
                check_raise(example, mtype_in, scitype_in)
            with pytest.raises(TypeError):
                check_raise(example, mtype_in)

        # test mtype
        assert mtype(example, scitype_in) == str(mtype_in)

        # test mtype to scitype
        assert mtype_to_scitype(mtype_in) == str(scitype_in)

        to_type = SeriesMtype.np_array
        if scitype_in is Scitype.panel:
            to_type = PanelMtype.np_3d_array
        assert isinstance(
            convert(example, from_type=mtype_in, to_type=to_type), np.ndarray
        )
        assert isinstance(convert_to(example, to_type=to_type), np.ndarray)


@pytest.mark.parametrize("mtype", ENUM_PANEL_MTYPES)
def test_panel_enums(mtype: PanelMtype) -> None:
    """Test panel enums.

    Parameters
    ----------
    mtype: PanelMtype
        Panel mtype to test
    """
    _run_test(mtype, Scitype.panel)


@pytest.mark.parametrize("mtype", ENUM_SERIES_MTYPES)
def test_series_enums(mtype: SeriesMtype) -> None:
    """Test series enum.

    Parameters
    ----------
    mtype: SeriesMtype
        Series mtype to test
    """
    _run_test(mtype, Scitype.series)
