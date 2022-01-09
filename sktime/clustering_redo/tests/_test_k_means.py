# -*- coding: utf-8 -*-
from sktime.clustering_redo._k_means import KMeans
from sktime.datasets import load_UCR_UEA_dataset

dataset_name = "Beef"


def test_kmeans():
    """Test implementation of Kmeans."""
    X_train, y_train = load_UCR_UEA_dataset(
        dataset_name, split="train", return_X_y=True
    )
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    kmeans = KMeans(random_state=1)
    kmeans.fit(X_train)
    # test = kmeans.predict(X_test)
    # from sklearn import metrics
    # score = metrics.rand_score(y_test, test)
