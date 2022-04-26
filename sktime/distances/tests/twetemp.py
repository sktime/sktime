#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:16:54 2020
@author: pfm
"""
import numpy as np
import numpy as np


def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1.0 / p)


def pytwed(A, timeSA, B, timeSB, nu, lmbda, degree):
    # [distance, DP] = TWED( A, timeSA, B, timeSB, lambda, nu )
    # Compute Time Warp Edit Distance (TWED) for given time series A and B
    #
    # A      := Time series A (e.g. [ 10 2 30 4])
    # timeSA := Time stamp of time series A (e.g. 1:4)
    # B      := Time series B
    # timeSB := Time stamp of time series B
    # lambda := Penalty for deletion operation
    # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # degree := Degree of the p norm for local cost.
    # Reference :
    #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    #    http://people.irisa.fr/Pierre-Francois.Marteau/

    n = A.shape[0]
    m = B.shape[0]
    # Dynamical programming
    DP = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    DP[0, 1:] = np.inf
    DP[1:, 0] = np.inf

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):

            # Deletion in A
            del_a = (
                DP[i - 1, j]
                + Dlp(A[i - 1], A[i], p=degree)
                + nu
                + lmbda
            )

            # Deletion in B
            del_b = (
                DP[i, j - 1]
                + Dlp(B[j - 1], B[j], p=degree)
                + nu
                + lmbda
            )

            # Keep data points in both time series
            match = (
                DP[i - 1, j - 1]
                + Dlp(A[i], B[j], p=degree)
                + Dlp(A[i - 1], B[j - 1], p=degree)
                + nu
            )

            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = min(del_a, del_b, match)

    distance = DP[n-1, m-1]
    return distance, DP

def backtracking(DP):
    # [ best_path ] = BACKTRACKING ( DP )
    # Compute the most cost efficient path
    # DP := DP matrix of the TWED function

    x = np.shape(DP)
    i = x[0] - 1
    j = x[1] - 1

    # The indices of the paths are save in opposite direction
    # path = np.ones((i + j, 2 )) * np.inf;
    best_path = []

    steps = 0
    while i != 0 or j != 0:
        best_path.append((i - 1, j - 1))

        C = np.ones((3, 1)) * np.inf

        # Keep data points in both time series
        C[0] = DP[i - 1, j - 1]
        # Deletion in A
        C[1] = DP[i - 1, j]
        # Deletion in B
        C[2] = DP[i, j - 1]

        # Find the index for the lowest cost
        idx = np.argmin(C)

        if idx == 0:
            # Keep data points in both time series
            i = i - 1
            j = j - 1
        elif idx == 1:
            # Deletion in A
            i = i - 1
            j = j
        else:
            # Deletion in B
            i = i
            j = j - 1
        steps = steps + 1

    best_path.append((i - 1, j - 1))

    best_path.reverse()
    return best_path[1:]


def twed(s1, s2, ts1=None, ts2=None, lmbda=1.0, nu=0.001, p=2, fast=True):
    """
    Time Warped Edit Distance (TWED)
    Parameters
    ----------
    s1 : np.ndarray
        First time series
    s2 : np.ndarray
        Second time series
    ts1 : np.ndarray, default: None
        Time stamps of first time series. If None, then equally spaced time stamps are assumed.
    ts2 : np.ndarray, default: None
        Time stamps of second time series. If None, then equally spaced time stamps are assumed.
    lmbda: float >= 0, default: 1.0
        A constant penalty that punishes the editing efforts
    nu: float > 0, default: 0.001
        A non-negative constant which characterizes the stiffness of the elastic TWED measure.
    p: int
        Order of the p-norm for local cost.
    fast: boolean, default: True
        If true, uses fast C implementation, if False uses Python reference implementation. Default "True" should usually be used, as it is typically magnitutes faster.
    Returns
    -------
    float
        twe distance
    """
    # Check if input arguments
    if (ts1 is not None) and (len(s1) != len(ts1)):
        raise ValueError("The length of s1 is not equal length of ts1.")

    if (ts2 is not None) and (len(s2) != len(ts2)):
        raise ValueError("The length of s2 is not equal length of ts2.")

    if ts1 is None:
        ts1 = np.arange(len(s1))

    if ts2 is None:
        ts2 = np.arange(len(s2))

    if nu <= 0.:
        raise ValueError("nu must be > 0.")

    if lmbda < 0:
        raise ValueError("lmbda must be >= 0.")

    if len(s1.shape) == 1:
        s1 = s1.reshape((-1, 1))

    if len(s2.shape) == 1:
        s2 = s2.reshape((-1, 1))

    _s1 = s1
    _s2 = s2
    s1 = np.vstack(([[0] * s1.shape[1]], s1))
    ts1 = np.hstack(([0], ts1))
    s2 = np.vstack(([[0] * s2.shape[1]], s2))
    ts2 = np.hstack(([0], ts2))

    return pytwed(A=s1, timeSA=ts1, B=s2, timeSB=ts2, nu=nu, lmbda=lmbda, degree=p)

