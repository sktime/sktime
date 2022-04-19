# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.datasets import load_acsf1, load_osuleaf


def run_exp(X_train):
    kmeans = TimeSeriesKMeans(
        averaging_method="dba",
        random_state=1,
        n_init=2,
        n_clusters=4,
        init_algorithm="kmeans++",
        metric="msm",
    )
    train_predict = kmeans.fit_predict(X_train)
    train_mean_score = metrics.rand_score(y_train, train_predict)

    print(train_mean_score)
    print(kmeans.inertia_)


if __name__ == "__main__":
    print("RUNNING")

    X_train, y_train = load_acsf1(split="train")
    X_test, y_test = load_acsf1(split="test")

    # run_exp(X_train)
    X_train, y_train = load_osuleaf(split="train")
    run_exp(X_train)

# dtw dba:dtw
# 0.5501010101010101
# 11760.83695034719
# 0.7144221105527638
# 3294.6395008389227

# msm dba:msm
# 0.636969696969697
# 15981.222524678658
# 0.1813065326633166
# 3643.4597794465803

# msm dba:dtw
# 0.6103030303030303
# 16118.294590592068
# 0.23020100502512564
# 3606.3261494388603
