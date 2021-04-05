""" DBScan time series clustering built on sklearn DBSCAN that
supports a range of distance measure specifically for time series. This distance
functions are defined in cython in sktime.distances.elastic_cython. Python versions
are in sktime.distances.elastic, but these are orders of magnitude slower.

"""



__author__ = "Josue Davalos"
__all__ = ["DBSCANTimeSeries"]


from sklearn.cluster import DBSCAN as _DBScan
from sktime.utils.validation.panel import check_X
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin 
from sklearn.utils.validation import check_array
from sktime.distances.elastic import euclidean_distance
from sktime.distances.elastic_cython import (
    ddtw_distance,
    dtw_distance,
    erp_distance,
    lcss_distance,
    msm_distance,
    twe_distance,
    wddtw_distance,
    wdtw_distance,
)


class DBSCANTimeSeries(_DBScan, ClusterMixin):
    """
    An adapted version of the scikit-learn DBSCAN to work with
    time series data.

    Necessary changes required for time series data:
        -   This however assumes that data must be 2d (a set of multivariate
            time series is 3d). Therefore these methods
            needed to be overridden to change this call to the following to
            support 3d data:
                n_samples = X.shape[0]
        -   check array has been disabled. This method allows nd data via an
            argument in the method header. However, there
            seems to be no way to set this in the cluster and allow it to
            propagate down to the method. Therefore, this
            method has been temporarily disabled (and then re-enabled). It
            is unclear how to fix this issue without either
            writing a new clustering from scratch or changing the
            scikit-learn implementation.


    Parameters
    ----------
    eps             : float, The maximum distance between two samples for one to be considered
    as in the neighborhood of the other. This is not a maximum bound
    on the distances of points within a cluster. This is the most
    important DBSCAN parameter to choose appropriately for your data set
    and distance function. (default =1)
    min_samples     : int, The number of samples (or total weight) in a neighborhood for a point
    to be considered as a core point. This includes the point itself. (default =5)
    weights         : mechanism for weighting a vote: 'uniform', 'distance'
    or a callable function: default ==' uniform'
    algorithm       : search method for neighbours {‘auto’,‘brute’}: 
    default = 'brute'
    metric          : distance measure for time series: {'dtw','ddtw',
    'wdtw','lcss','erp','msm','twe'}: default ='euclidean'

    """
    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        weights="uniform",
        algorithm="brute",
        metric="euclidean",
        **kwargs
    ):
        if algorithm == "kd_tree":
            raise ValueError(
                "DBSCANTimeSeries cannot work with kd_tree since kd_tree "
                "cannot be used with a callable distance metric and we do not support "
                "precalculated distances as yet."
            )
        if algorithm == "ball_tree":
            raise ValueError(
                "DBSCANTimeSeries cannot work with ball_tree since "
                "ball_tree has a list of hard coded distances it can use, and cannot "
                "work with 3-D arrays"
            )
        if metric == "euclidean":  # Euclidean will default to the base class distance
            metric = euclidean_distance
        if metric == "dtw":
            metric = dtw_distance
        elif metric == "ddtw":
            metric = ddtw_distance
        elif metric == "wdtw":
            metric = wdtw_distance
        elif metric == "wddtw":
            metric = wddtw_distance
        elif metric == "lcss":
            metric = lcss_distance
        elif metric == "erp":
            metric = erp_distance
        elif metric == "msm":
            metric = msm_distance
        elif metric == "twe":
            metric = twe_distance
        elif metric == "mpdist":
            metric = mpdist
            # When mpdist is used, the subsequence length (parameter m) must be set
            # Example: knn_mpdist = KNeighborsTimeSeriesClassifier(
            # metric='mpdist', metric_params={'m':30})
        else:
            if type(metric) is str:
                raise ValueError(
                    "Unrecognised distance measure: " + metric + ". Allowed values "
                    "are names from [euclidean,dtw,ddtw,wdtw,wddtw,lcss,erp,msm] or "
                    "please pass a callable distance measure into the constuctor"
                )

        super(DBSCANTimeSeries, self).__init__(
            eps = eps,
            algorithm=algorithm,
            min_samples = min_samples,
            metric=metric,
            **kwargs
        )
        self._is_fitted = False



    def fit(self, X, y=None):
        """Fit the model using X as training data 

        Parameters
        ----------
        X : sktime-format pandas dataframe with shape([n_cases,n_dimensions]),
        or numpy ndarray with shape([n_cases,n_readings,n_dimensions])
        """
        X = check_X(X, enforce_univariate=False, coerce_to_numpy=True)        
        
        if hasattr(check_array, "__wrapped__"):
            temp = check_array.__wrapped__.__code__
            check_array.__wrapped__.__code__ = _check_array_ts.__code__
        else:
            temp = check_array.__code__
            check_array.__code__ = _check_array_ts.__code__
        
        fx = super().fit(X)
        self._is_fitted = True
        return fx
    

# overwrite sklearn internal checks, this is really hacky
# we now need to replace: check_array.__wrapped__.__code__ since it's
# wrapped by a future warning decorator
def _check_array_ts(array, *args, **kwargs):
    return array