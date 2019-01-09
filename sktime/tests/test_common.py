import pytest

from sklearn.utils.estimator_checks import check_estimator

from sktime import TSDummyClassifier


@pytest.mark.parametrize(
    "Estimator", [TSDummyClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
