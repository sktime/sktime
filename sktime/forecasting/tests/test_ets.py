__author__ = ["Hongyi Yang"]

import numpy as np
from numpy.testing import assert_array_equal
from sktime.forecasting.ets import AutoETS
from sktime.datasets import load_airline
from rpy2.robjects import r
from rpy2.robjects.packages import importr

y_train = load_airline()


# Test _pegelsresid function against R internal pegelsresid.C function
# by passing a simple set of input arguments
def test_pegelsresid():
    # Load R package
    forecast = importr('forecast')
    y = y_train.tolist()
    # Get result from pegelsresid.C function
    res_R = forecast.pegelsresid_C(y, 12, 0, "A", "N", "N", False, 0.5,
                                   r("NULL"), r("NULL"), r("NULL"), 3)
    lik_R = np.array(list(res_R[0]))
    amse_R = np.array(list(res_R[1]))
    e_R = np.array(list(res_R[2]))
    states_R = np.array(list(res_R[3])).reshape(-1, 1)

    # Calculate result from _pegelsresid function
    ets = AutoETS()
    lik_ets, amse_ets, e_ets, states_ets = ets._pegelsresid(y, 12, [0], "A",
                                                            "N", "N", False,
                                                            0.5, None, None,
                                                            None, 3)
    lik_ets = np.array(lik_ets)
    assert_array_equal(lik_ets, lik_R)
    assert_array_equal(amse_ets, amse_R)
    assert_array_equal(e_ets, e_R)
    assert_array_equal(states_ets, states_R)
