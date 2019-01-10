import pytest

from sklearn.utils.estimator_checks import check_estimator

from sktime import TSDummyClassifier
from sktime import TSDummyRegressor


@pytest.mark.parametrize(
    "Estimator", [TSDummyClassifier, TSDummyRegressor]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
