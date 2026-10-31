# -*- coding: utf-8 -*-
"""Test Next generation reservoir computing forecasters."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["frthjf"]

import numpy as np
import sklearn.linear_model
from scipy.integrate import solve_ivp
from sktime.performance_metrics import mean_squared_error

from sktime.forecasting.reservoir import NextRC


def test_next_rc_estimator():

    # --- generate test data

    warmup = 1.0
    dt = 0.25
    traintime = 100.0
    ridge_param = 1.0e-3
    testtime = 800.0
    maxtime = warmup + traintime + testtime
    warmup_pts = round(warmup / dt)
    traintime_pts = round(traintime / dt)
    warmtrain_pts = warmup_pts + traintime_pts
    testtime_pts = round(testtime / dt)
    maxtime_pts = round(maxtime / dt)
    # lyaptime_pts=round(lyaptime/dt)
    # input dimension
    d = 3
    # number of time delay taps
    k = 2
    # size of the linear part of the feature vector
    dlin = k * d
    # size of the nonlinear part of feature vector
    dnonlin = int(dlin * (dlin + 1) * (dlin + 2) / 6)
    # total size of feature vector: linear + nonlinear
    dtot = dlin + dnonlin
    # t values for whole evaluation time
    # (need maxtime_pts + 1 to ensure a step of dt)
    t_eval = np.linspace(0, maxtime, maxtime_pts + 1)

    r1 = 1.2
    r2 = 3.44
    r4 = 0.193
    alpha = 11.6
    ir = 2 * 2.25e-5  # the 2 is for the hyperbolic sine

    def doublescroll(t, y):
        # y[0] = V1, y[1] = V2, y[2] = I
        dV = y[0] - y[1]  # V1-V2
        g = (dV / r2) + ir * np.sinh(alpha * dV)
        dy0 = (y[0] / r1) - g
        dy1 = g - y[2]
        dy2 = y[1] - r4 * y[2]

        return [dy0, dy1, dy2]

    # I integrated out to t=50 to find points on the attractor, then use these as the initial conditions
    doublescroll_soln = solve_ivp(
        doublescroll,
        (0, maxtime),
        [0.37926545, 0.058339, -0.08167691],
        t_eval=t_eval,
        method="RK23",
    )

    # --- manual prediction

    # create an array to hold the linear part of the feature vector
    x = np.zeros((dlin, maxtime_pts))

    # fill in the linear part of the feature vector for all times
    for delay in range(k):
        for j in range(delay, maxtime_pts):
            x[d * delay : d * (delay + 1), j] = doublescroll_soln.y[:, j - delay]

    # create an array to hold the full feature vector for training time
    out_train = np.zeros((dtot, traintime_pts))

    # copy over the linear part
    out_train[0:dlin, :] = x[:, warmup_pts - 1 : warmtrain_pts - 1]
    # raise ValueError(x[:,warmup_pts-1:warmtrain_pts-1])
    # fill in the non-linear part
    cnt = 0
    for row in range(dlin):
        for column in range(row, dlin):
            for span in range(column, dlin):
                out_train[dlin + cnt] = (
                    x[row, warmup_pts - 1 : warmtrain_pts - 1]
                    * x[column, warmup_pts - 1 : warmtrain_pts - 1]
                    * x[span, warmup_pts - 1 : warmtrain_pts - 1]
                )
                cnt += 1

    W_out = (
        (x[0:d, warmup_pts:warmtrain_pts] - x[0:d, warmup_pts - 1 : warmtrain_pts - 1])
        @ out_train[:, :].T
        @ np.linalg.pinv(
            out_train[:, :] @ out_train[:, :].T + ridge_param * np.identity(dtot)
        )
    )
    x_predict = (
        x[0:d, warmup_pts - 1 : warmtrain_pts - 1]
        + W_out @ out_train[:, 0:traintime_pts]
    )

    # create a place to store feature vectors for prediction
    out_test = np.zeros(dtot)  # full feature vector
    x_test = np.zeros((dlin, testtime_pts))  # linear part

    # copy over initial linear feature vector
    x_test[:, 0] = x[:, warmtrain_pts - 1]
    # do prediction
    for j in range(testtime_pts - 1):
        # copy linear part into whole feature vector
        out_test[0:dlin] = x_test[:, j]
        # fill in the non-linear part
        cnt = 0
        for row in range(dlin):  # x
            for column in range(row, dlin):  # y
                for span in range(column, dlin):  # z
                    out_test[dlin + cnt] = (
                        x_test[row, j] * x_test[column, j] * x_test[span, j]
                    )
                    cnt += 1
        # fill in the delay taps of the next state
        x_test[d:dlin, j + 1] = x_test[0 : (dlin - d), j]
        # do a prediction
        x_test[0:d, j + 1] = x_test[0:d, j] + W_out @ out_test[:]

    #  ---- assert match with estimator implementation

    rc = NextRC(regressor=sklearn.linear_model.Ridge(alpha=1.0e-3, fit_intercept=False))
    X = doublescroll_soln.y.T[0:warmtrain_pts, :]
    rc.fit(X, None, fh=[warmup_pts - 1, warmtrain_pts - 1])

    # matching ridge regression
    np.testing.assert_allclose(W_out, rc.regressor.coef_, rtol=0.1)

    # matching training predictions
    x_predict_t = rc.predict(X, fh=[warmup_pts - 1, warmtrain_pts - 1])
    np.testing.assert_allclose(x_predict, x_predict_t)

    # matching test predictions
    X_test = doublescroll_soln.y.T[warmtrain_pts - 2 : warmtrain_pts, :d]
    for j in range(100):
        y_test = rc.predict(X_test, fh=[1, 2])
        np.testing.assert_allclose(x_test[:d, 1 + j : 2 + j], y_test, rtol=0.05)
        # auto-regressive feeding
        X_test[0] = X_test[1]
        X_test[1] = y_test.T[0]

    # x-y training
    X = doublescroll_soln.y.T[0:warmtrain_pts, 0:2]
    y = doublescroll_soln.y.T[0:warmtrain_pts, 2:3]

    rc.fit(X, y, fh=[warmup_pts - 1, warmtrain_pts - 1])
    y_pred = rc.predict(X, fh=[warmup_pts - 1, warmtrain_pts - 1])
    err = mean_squared_error(y[warmup_pts - 1 : warmtrain_pts - 1, :], y_pred)
    assert err < 3e-5

    # test load and store
    state = rc.__getstate__()
    rc2 = NextRC()
    rc2.__setstate__(state)
    y_pred2 = rc.predict(X, fh=[warmup_pts - 1, warmtrain_pts - 1])
    np.testing.assert_allclose(y_pred2, y_pred, rtol=0.05)
    assert rc2.d_linear_ == rc.d_linear_
