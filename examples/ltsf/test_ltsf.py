from sktime.forecasting.ltsf import LTSFLinearForecaster
from sktime.utils.estimator_checks import check_estimator

res = check_estimator(
    LTSFLinearForecaster,
    tests_to_run=["test_hierarchical_with_exogeneous"],
    raise_exceptions=True,
)
