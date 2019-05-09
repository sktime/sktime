from sktime.classifiers.elastic_ensemble import ElasticEnsemble
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.classifiers.time_series_neighbors import KNeighborsTimeSeriesClassifier

from sktime.utils.estimator_checks import check_ts_estimator

#TODO change test so that all classes in enumerated packages are tested for conformity

def test_elastic_ensemble():
        return check_ts_estimator(ElasticEnsemble())

def test_time_series_forest_classifier():
        return check_ts_estimator(TimeSeriesForestClassifier())

def test_k_neighbours_timesries_classifier():
        return check_ts_estimator(KNeighborsTimeSeriesClassifier())
