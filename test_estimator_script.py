from sktime.forecasting.base.adapters._autots import _AutoTSAdapter
from sktime.utils.estimator_checks import check_estimator

#
# check_estimator(
#     _AutoTSAdapter,
#     raise_exceptions=True,
#     tests_to_run="test_predict_time_index_with_X",
# )
check_estimator(_AutoTSAdapter, raise_exceptions=False, tests_to_run=None)
