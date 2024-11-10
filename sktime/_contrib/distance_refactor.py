"""Test the move from (m,d) to (d,m)."""

import numpy as np
from sklearn import metrics

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.datasets import load_arrow_head, load_basic_motions, load_unit_test
from sktime.distances import dtw_distance, erp_distance, lcss_distance
from sktime.distances.tests._utils import create_test_distance_numpy

# Clustering With num custers set to 2 and transpose
expected_rand_unit_test = {
    "euclidean": 0.5210526315789473,
    "dtw": 0.5210526315789473,
    "erp": 0.47368421052631576,
    "edr": 0.4789473684210526,
    "wdtw": 0.5210526315789473,
    "lcss": 0.7315789473684211,
    "msm": 0.6052631578947368,
    "ddtw": 0.6631578947368421,
    "wddtw": 0.6052631578947368,
}
expected_rand_basic_motions = {
    "euclidean": 0.5947368421052631,
    "dtw": 0.5421052631578948,
    "erp": 0.47368421052631576,
    "edr": 0.4842105263157895,
    "wdtw": 0.5947368421052631,
    "lcss": 0.7,
    "msm": 0.7947368421052632,
    "ddtw": 0.5736842105263158,
    "wddtw": 0.6684210526315789,
}

distances = [
    "euclidean",
    "dtw",
    "wdtw",
    "erp",
    "edr",
    "lcss",
    "msm",
    "ddtw",
    "wddtw",
]
from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm


def time_clusterers():
    """Time tests for clusterers."""
    k_means = TimeSeriesKMeans(
        n_clusters=5,  # Number of desired centers
        init_algorithm="forgy",  # Center initialisation technique
        max_iter=10,  # Maximum number of iterations for refinement on training set
        metric="dtw",  # Distance metric to use
        averaging_method="mean",  # Averaging technique to use
        random_state=1,
    )
    X_train, y_train = load_arrow_head(split="train")
    X_test, y_test = load_arrow_head(split="test")
    k_means.fit(X_train)
    plot_cluster_algorithm(k_means, X_test, k_means.n_clusters)


def debug_clusterers():
    """Debug clusterers."""
    X_train, y_train = load_basic_motions(split="train", return_type="numpy3d")
    #    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    #   X_train2, y_train2 = load_unit_test(split="train", return_type="numpy2d")
    parameters = {"window": 1.0, "epsilon": 50.0, "g": 0.05, "c": 1.0}
    for dist in distances:
        kmeans = TimeSeriesKMeans(
            averaging_method="mean",
            random_state=1,
            n_init=2,
            n_clusters=2,
            init_algorithm="kmeans++",
            metric=dist,
            distance_params=parameters,
        )
        kmeans.fit(X_train)
        y_pred = kmeans.predict(X_train)
        train_rand = metrics.rand_score(y_train, y_pred)
        print('"' + dist + '": ' + str(train_rand) + ",")
        # kmeans.fit(X_train2)
        # y_pred = kmeans.predict(X_train2)
        # train_rand2 = metrics.rand_score(y_train2, y_pred)
        # assert train_rand == train_rand2
        # print(" Rand score on with 2D unit test = ",train_rand)


from sklearn.preprocessing import StandardScaler


def generate_test_results_clusterers():
    """Generate test results."""
    X_train, y_train = load_unit_test(split="train", return_type="numpy2d")
    print(" Shape = ", X_train[0].shape)
    s = StandardScaler()
    X_train = s.fit_transform(X_train.T)
    X_train = X_train.T
    d1 = X_train[0]
    d2 = X_train[1]
    print(" Shape = ", d1.shape)
    d3 = np.zeros(d1.shape)
    print(d1)
    print(d2)
    print(d3)
    # d1, c1 and c2= distance =  0.9166666666666666

    c1 = np.array(
        [
            -0.70310574,
            -0.87214184,
            -1.00698691,
            -1.06674154,
            -1.12436733,
            -1.15538656,
            -1.15125921,
            -1.03439438,
            -0.85569435,
            -0.76934041,
            -0.47880807,
            0.13013539,
            1.24150173,
            1.47580796,
            1.21524191,
            1.03776906,
            0.9338891,
            1.11147877,
            1.35705589,
            1.29823615,
            0.81583304,
            0.30507612,
            -0.14019819,
            -0.5636006,
        ]
    )
    c2 = np.array(
        [
            -0.56887401,
            -0.78355128,
            -0.97270354,
            -1.05610923,
            -1.16663863,
            -1.21763814,
            -1.21428503,
            -1.14248176,
            -0.9842895,
            -0.91458795,
            -0.55085488,
            0.03584731,
            1.08807594,
            1.32503268,
            1.08481383,
            0.90703813,
            0.87781706,
            1.03376858,
            1.33278512,
            1.38938799,
            0.99061415,
            0.59018919,
            0.1635356,
            -0.24689165,
        ]
    )
    x = create_test_distance_numpy(10)
    y = create_test_distance_numpy(10, random_state=2)
    curr_X = X_train[19]

    curr_X = curr_X.reshape((1, c1.shape[0]))

    c1 = c1.reshape((1, c1.shape[0]))
    print(f"instance 19  = {curr_X} shape X = {curr_X.shape}")
    print("Centroid 0 =", c1)
    d = lcss_distance(curr_X, c1, epsilon=0.01)
    print(" DISTANCE = ", d)
    print("Shape centroid = ", c1.shape)
    d = lcss_distance(curr_X, c2, epsilon=0.01)
    print(" DISTANCE = ", d)
    print("Shape centroid = ", c2.shape)

    italy1 = np.array(
        [
            -0.71051757,
            -1.1833204,
            -1.3724416,
            -1.5930829,
            -1.4670021,
            -1.3724416,
            -1.0887599,
            0.045966947,
            0.92853223,
            1.0861332,
            1.2752543,
            0.96005242,
            0.61333034,
            0.014446758,
            -0.6474772,
            -0.26923494,
            -0.20619456,
            0.61333034,
            1.3698149,
            1.4643754,
            1.054613,
            0.58181015,
            0.1720477,
            -0.26923494,
        ]
    )
    italy2 = np.array(
        [
            -0.41137184,
            -0.67348487,
            -1.1103399,
            -1.2413964,
            -1.4161385,
            -1.3287674,
            -0.84822689,
            -0.58611386,
            -0.80454138,
            0.069168714,
            0.68076578,
            0.76813679,
            0.63708028,
            0.069168714,
            -0.36768633,
            -0.49874285,
            -0.36768633,
            -0.018202296,
            1.7729034,
            2.0350164,
            1.5544759,
            1.4234194,
            0.63708028,
            0.025483209,
        ]
    )
    d = dtw_distance(italy1, italy2)
    print(" Italy DTW DISTANCE = ", d)


def difference_test():
    """Test the distance functions with allowable input.

    TEST 1: Distances. Generate all distances with tsml, compare.
    TEST 2: Classification.
    TEST 3: Clustering.
    TEST 4: tslearn
    """
    X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    d1 = X_train[0]
    d2 = X_train[2]
    dist = erp_distance
    #    d1=np.transpose(d1)
    #    d2=np.transpose(d2)
    print("Shape  = ", d1.shape)
    name = "LCSS"
    no_window = np.zeros((d1.shape[1], d2.shape[1]))
    # "wi [0.0, 0.1, 1.0],  # window
    dist1 = dist(d1, d2, window=0.0)
    print(name, " w = 0 dist = ", dist1)
    dist1 = dist(d1, d2, window=0.1)
    print(name, " w = 0.1. dist 1 = ", dist1)
    dist1 = dist(d1, d2, window=1.0)
    print(name, " w = 1 dist 1 = ", dist1)
    print(" SHAPE  = ", d1.shape)
    X_train, y_train = load_basic_motions(split="train", return_type="numpy3d")
    b1 = X_train[0]
    b2 = X_train[1]
    #    b1 = np.transpose(b1)
    #    b2 = np.transpose(b2)
    dist = dtw_distance
    print("BM shape = ", b1.shape)
    dist2 = dist(b1, b2, epsilon=0.0)
    print(" g = 0.0, BASIC MOTIONS DIST = ", dist2)
    dist2 = dist(b1, b2, epsilon=1.0)
    print(" g = 0.1, BASIC MOTIONS DIST = ", dist2)
    dist2 = dist(b1, b2, epsilon=4.0)
    print(" g = 1, BASIC MOTIONS DIST = ", dist2)


#   dist2 = euclidean_distance(b1, b2)
#   print(" ED BASIC MOTIONS DIST = ", dist2)


#  print(" Window = 1, BASIC MOTIONS DIST = ", dist2)


if __name__ == "__main__":
    #    debug_clusterers()
    # difference_test()
    # time_clusterers()
    generate_test_results_clusterers()
