import operator

import numpy as np
from pandas import Series
from pandas.tests.extension import base as extension_tests

from sktime.container import TimeDtype, TimeArray
from sktime.container.array import from_list

import pytest



# pytest Fixtures --------------------------------------------------------------

@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return TimeDtype()

def make_data():
    a = np.array([[i + j for j in range(10)] for i in range(100)],
                 dtype=np.float)
    ta = TimeArray(a)
    return ta

@pytest.fixture
def data():
    """
    Length-100 array for this type.
    * data[0] and data[1] should both be non missing
    * data[0] and data[1] should not be equal
    """
    return make_data()


@pytest.fixture
def data_for_twos():
    """Length-100 array in which all the elements are two."""
    raise NotImplementedError


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return from_list([[[np.nan, np.nan], [np.nan, np.nan]],
                      [[1., 2.], [0., 1.]]])


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.
    Parameters
    ----------
    data : fixture implementing `data`
    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    """
    Length-3 array with a known sort order.
    This should be three items [B, C, A] with
    A < B < C
    """
    raise NotImplementedError


@pytest.fixture
def data_missing_for_sorting():
    """
    Length-3 array with a known sort order.
    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise NotImplementedError


@pytest.fixture
def na_cmp():
    """
    Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.is_``
    """
    return operator.is_


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return None


@pytest.fixture
def data_for_grouping():
    """
    Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    raise NotImplementedError


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param

_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    # '__sub__', '__rsub__',
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations
    Adapted to excluse __sub__, as this is implemented as "difference".
    """
    return request.param

# ------------------------------------------------------------------------------
# Inherited tests
# ------------------------------------------------------------------------------

class TestDtype(extension_tests.BaseDtypeTests):
    pass

class TestConstructors(extension_tests.BaseConstructorsTests):
    pass

class TestGetitem(extension_tests.BaseGetitemTests):
    pass

class TestSetitem(extension_tests.BaseSetitemTests):
    pass

class TestInterface(extension_tests.BaseInterfaceTests):
    pass

class TestMissing(extension_tests.BaseMissingTests):
    pass

class TestCasting(extension_tests.BaseCastingTests):
    pass

class TestPrinting(extension_tests.BasePrintingTests):
    pass


# Under current development
#class TestArithmeticOps(extension_tests.BaseArithmeticOpsTests):
#    pass

#class TestMethods(extension_tests.BaseMethodsTests):
#    pass

# TODO: implement _from_sequence_of_strings
#class TestParsing(extension_tests.BaseParsingTests):
#    pass


# See here for more tests:
# https://github.com/geopandas/geopandas/blob/master/geopandas/tests/test_extension_array.py