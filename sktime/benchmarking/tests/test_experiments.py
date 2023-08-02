"""Functions to test the functions in experiments.py."""
import pytest

from sktime.benchmarking.experiments import (
    run_classification_experiment,
    run_clustering_experiment,
)
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.clustering.k_means import TimeSeriesKMeans
from sktime.datasets import load_unit_test
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_run_clustering_experiment(tmp_path):
    """Test running and saving results for clustering.

    Currently it just checks the files have been created, then deletes them.
    """
    dataset = "UnitTest"
    train_X, train_Y = load_unit_test("TRAIN")
    test_X, test_Y = load_unit_test("TEST")
    run_clustering_experiment(
        train_X,
        TimeSeriesKMeans(n_clusters=2),
        results_path=tmp_path,
        trainY=train_Y,
        testX=test_X,
        testY=test_Y,
        cls_name="kmeans",
        dataset_name=dataset,
        resample_id=0,
    )
    test_path = tmp_path.joinpath(f"kmeans/Predictions/{dataset}/testResample0.csv")
    train_path = tmp_path.joinpath(f"kmeans/Predictions/{dataset}/trainResample0.csv")
    assert test_path.is_file()
    assert train_path.is_file()
    # remove files
    test_path.unlink()
    train_path.unlink()


@pytest.mark.skipif(
    not _check_soft_dependencies("numba", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_run_classification_experiment(tmp_path):
    """Test running and saving results for classifiers.

    Currently it just checks the files have been created, then deletes them.
    """
    dataset = "UnitTest"
    train_X, train_Y = load_unit_test("TRAIN")
    test_X, test_Y = load_unit_test("TEST")
    run_classification_experiment(
        train_X,
        train_Y,
        test_X,
        test_Y,
        TimeSeriesForestClassifier(n_estimators=10),
        str(tmp_path),
        cls_name="TSF",
        dataset="UnitTest",
        resample_id=0,
        train_file=True,
    )
    test_path = tmp_path.joinpath(f"TSF/Predictions/{dataset}/testResample0.csv")
    train_path = tmp_path.joinpath(f"TSF/Predictions/{dataset}/trainResample0.csv")
    assert test_path.is_file()
    assert train_path.is_file()
    # remove files
    test_path.unlink()
    train_path.unlink()
