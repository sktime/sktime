import numpy as np
from sktime.utils.seasonality import remove_seasonality
from sktime.utils.seasonality import add_seasonality
import pytest


@pytest.mark.parametrize("freq", [1, 4, 7, 12, 365])
def test_remove_add_seasonality(freq):

    # generate random data with some seasonality
    n = np.random.randint(freq * 2, freq * 4)
    s = np.zeros(n)
    s[::freq] = 20
    y = np.random.lognormal(size=n) + s

    # check if removing and addding it back returns the same data
    yt, si = remove_seasonality(y, freq)
    yit = add_seasonality(yt, si)
    np.testing.assert_almost_equal(y, yit)
