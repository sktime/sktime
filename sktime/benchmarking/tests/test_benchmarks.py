from sklearn.ensemble import RandomForestClassifier
from sktime.benchmarking.benchmarks import BaseBenchmark


def test_sklearn_estimator_can_be_added():
    bench = BaseBenchmark()
    clf = RandomForestClassifier()
    bench.add_estimator(clf)

    assert "RandomForestClassifier" in bench.estimators.entities
