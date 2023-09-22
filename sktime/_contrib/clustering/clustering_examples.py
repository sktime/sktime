"""Clustering usage tests and examples."""
import numpy as np

from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.clustering.k_medoids import TimeSeriesKMedoids
from sktime.datasets import load_arrow_head


def form_cluster_list(clusters, n) -> np.array:
    """Form a cluster list."""
    preds = np.zeros(n)
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            preds[clusters[i][j]] = i
    return preds


if __name__ == "__main__":
    clusterer1 = TimeSeriesKMeans(n_clusters=5, max_iter=50, averaging_algorithm="mean")
    clusterer2 = TimeSeriesKMedoids()
    X, y = load_arrow_head(return_X_y=True)
    clusterer1.fit(X)
    c = clusterer1.predict(X)
    x = form_cluster_list(c, len(y))
    for i in range(len(x)):
        print(i, " is in cluster ", x[i])
