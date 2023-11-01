from sktime.forecasting.ltsf import LTSFLinearForecaster
from sktime.utils.estimator_checks import check_estimator

res = check_estimator(
    LTSFLinearForecaster,
    tests_to_exclude=["test_predict_time_index_in_sample_full"],
    raise_exceptions=True,
)
