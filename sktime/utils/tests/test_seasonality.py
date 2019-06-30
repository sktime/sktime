import numpy as np
from sktime.utils.seasonality import remove_seasonality
from sktime.utils.seasonality import add_seasonality
import pytest


@pytest.mark.parametrize("model", ['additive', 'multiplicative'])
@pytest.mark.parametrize("sp", [1, 4, 7, 12, 365])
def test_remove_add_seasonality(sp, model):

    # generate random data with some seasonality
    n = np.random.randint(sp * 2, sp * 4)
    s = np.zeros(n)
    s[::sp] = 20
    y = np.random.lognormal(size=n) + s

    # check if removing and adding it back returns the same data
    yt, si = remove_seasonality(y, sp=sp, model=model)
    yit = add_seasonality(yt, si)
    np.testing.assert_almost_equal(y, yit)
