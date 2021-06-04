# -*- coding: utf-8 -*-
"""Clustering usage tests and examples"""


def test_loading():
    # test multivariate
    # Test univariate
    data_dir = "E:/tsc_ts/"
    dataset = "Gunpoint"
    trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
    testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
    print("Loaded " + dataset + " in position " + str(i))


if __name__ == "__main__":
    """
    Example simple usage, with arguments input via script or hard coded for testing
    """

