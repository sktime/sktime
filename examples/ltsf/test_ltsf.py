from sktime.forecasting.ltsf import LTSFLinearForecaster
from sktime.utils.estimator_checks import check_estimator

res = check_estimator(
    LTSFLinearForecaster,
    raise_exceptions=True,
)
