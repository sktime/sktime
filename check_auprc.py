from sktime.performance_metrics.detection import TimeSeriesAUPRC
from sktime.utils.estimator_checks import check_estimator

# Run all checks for the metric
check_estimator(TimeSeriesAUPRC)
