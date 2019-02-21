import pytest

from ..utils.estimator_checks import check_ts_estimator

from sktime.classifiers.example_classifiers import TSDummyClassifier
from sktime.regressors.example_regressors import TSDummyRegressor


@pytest.mark.parametrize(
    "Estimator", [TSDummyClassifier, TSDummyRegressor]
)
def test_all_estimators(Estimator):
    return check_ts_estimator(Estimator)
