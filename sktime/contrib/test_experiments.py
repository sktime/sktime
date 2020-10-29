""" test_experiments.py:

runs through all classifiers to build Chinatown
"""


def test_all_classifiers():
    data_dir = "../datasets/data/"
    dataset = "Chinatown"
    trainX, trainY = load_ts(data_dir + dataset + "/" + dataset + "_TRAIN.ts")
    testX, testY = load_ts(data_dir + dataset + "/" + dataset + "_TEST.ts")
