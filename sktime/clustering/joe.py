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
    # print("RUNNING")
    #
    # X_train, y_train = load_acsf1(split="train")
    # X_test, y_test = load_acsf1(split="test")

    # run_exp(X_train)
    # X_train, y_train = load_osuleaf(split="train")
    # run_exp(X_train)

    from sktime.datasets import load_UCR_UEA_dataset

    import matplotlib.pyplot as plt
    from sktime.clustering.metrics.averaging import dba
    from sktime.datatypes import convert_to
    from tslearn.barycenters import \
        euclidean_barycenter, \
        dtw_barycenter_averaging, \
        dtw_barycenter_averaging_subgradient, \
        softdtw_barycenter
    from tslearn.datasets import CachedDatasets

    X_train = load_UCR_UEA_dataset('Trace', split='train')[0]
    skX_train = convert_to(X_train, 'numpy3D')

    np.random.seed(0)
    X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")
    X = X_train[y_train == 2]
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    length_of_sequence = X.shape[1]

    X = X_train
    length_of_sequence = X.shape[1]


    def plot_helper(barycenter):
        # plot all points of the data set
        for series in X:
            plt.plot(series.ravel(), "k-", alpha=.2)
        # plot the given barycenter of them
        plt.plot(barycenter.ravel(), "r-", linewidth=2)

    # plot the four variants with the same number of iterations and a tolerance of
    # 1e-3 where applicable

    ax1 = plt.subplot()
    plt.subplot(sharex=ax1)
    plt.title("DBA (vectorized version of Petitjean's EM)")
    plot_helper(dba(X))

    # clip the axes for better readability
    ax1.set_xlim([0, length_of_sequence])

    # show the plot(s)
    plt.tight_layout()
    plt.show()

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