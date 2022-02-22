# -*- coding: utf-8 -*-
"""Testing utilities in the datatype module."""

__author__ = ["fkiraly"]

import pandas as pd
import pytest

from sktime.datatypes._examples import get_examples
from sktime.datatypes._utilities import get_cutoff

SCITYPE_MTYPE_PAIRS = [
    ("Series", "pd.Series"),
    ("Series", "pd.DataFrame"),
    ("Series", "np.ndarray"),
    ("Panel", "pd-multiindex"),
    ("Panel", "numpy3D"),
    ("Panel", "nested_univ"),
    ("Panel", "df-list"),
    ("Hierarchical", "pd_multiindex_hier"),
]


@pytest.mark.parametrize("scitype,mtype", SCITYPE_MTYPE_PAIRS)
def test_get_cutoff(scitype, mtype):
    """Tests that conversions for scitype agree with from/to example fixtures.

    Parameters
    ----------
    scitype : str - scitype of input
    mtype : str - mtype of input

    Raises
    ------
    AssertionError if get_cutoff does not return a length 1 pandas.index
        for any fixture example of given scitype, mtype
    """
    # retrieve example fixture
    fixtures = get_examples(mtype=mtype, as_scitype=scitype, return_lossy=False)

    for fixture in fixtures.values():
        if fixture is None:
            continue

        cutoff = get_cutoff(fixture)

        assert isinstance(cutoff, pd.Index)
        assert len(cutoff) == 1
