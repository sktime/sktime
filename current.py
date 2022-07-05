from tabnanny import check
from sktime.regression.deep_learning.cnn import CNNRegressor
from sktime.utils.estimator_checks import check_estimator
from icecream import ic

# a = check_estimator(CNNRegressor)
a = check_estimator(CNNRegressor, return_exceptions=False, fixtures_to_run='test_fit_idempotent[CNNRegressor-ClassifierFitPredictMultivariate]')
ic(a)
