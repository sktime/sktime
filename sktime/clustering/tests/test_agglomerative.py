from sktime.clustering.agglomerative import TimeSeriesAgglomerativeClustering
from sktime.datasets import load_basic_motions


def test_agglomerative_basic_run():
    """Test that agglomerative clustering runs without error on simple data."""
    X, _ = load_basic_motions(split="train")
    X = X[:10]

    clst = TimeSeriesAgglomerativeClustering(n_clusters=2, distance="dtw")
    clst.fit(X)

    assert hasattr(clst, "labels_")
    assert len(clst.labels_) == 10
    assert hasattr(clst, "linkage_matrix_")
    assert clst.linkage_matrix_.shape[0] == 9
