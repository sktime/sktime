# -*- coding: utf-8 -*-
from typing import Callable, Union
from sktime.clustering.utils import Numpy_Array
from sktime.distances.elastic_cython import ddtw_distance
from sktime.distances.elastic_cython import dtw_distance
from sktime.distances.elastic_cython import erp_distance
from sktime.distances.elastic_cython import lcss_distance
from sktime.distances.elastic_cython import msm_distance
from sktime.distances.elastic_cython import twe_distance
from sktime.distances.elastic_cython import wddtw_distance
from sktime.distances.elastic_cython import wdtw_distance
from sktime.distances.mpdist import mpdist

distance_function = Callable[[Numpy_Array, Numpy_Array, float], Numpy_Array]
distance_parameter = Union[distance_function, str]


class Cluster:
    """
    Base class used for clustering models. It is designed to deal with all
    the underlying sktime specific details so that new implementations can
    be used done quickly and efficiently. Additionally it seeks to make the
    use of the model as similar to the orginal design as possible

    Attributes
    ----------
    distance: distance_function
        function that is of the for (numpy_arr1, numpy_arr2) -> numpy_arr
        This should be used to specify a custom distance measure for given
        clusterers that allow it
    """

    def __init__(self, distance: distance_parameter) -> None:
        """
        Consturctor for a cluster algorithm.

        Parameters
        ----------
        distance: distance_parameter (distance_function | str)
            Distance function to be used in the clustering
            algorithm
        """
        self.distance = distance

    def fit(self, x, y):
        """
        Fit is a method that fits the given model
        """
        raise NotImplementedError("abstract method")

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample
        """
        raise NotImplementedError("abstract method")

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Computer clustering and transform X to cluster-distance space
        """
        raise NotImplementedError("abstract method")

    def predict(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to
        """
        raise NotImplementedError("abstract method")

    def score(self, y=None, sample_weight=None):
        """
        Opposite of the value of X
        """
        raise NotImplementedError("abstract method")

    def __compute_distance(self):
        """
        Method that is called to compute the distance
        """
        pass

    @distance_function.setter
    def __set_distance_function(self, distance: distance_parameter) -> None:
        """
        Setter method for the distance_function property

        Parameters
        ----------
        distance: distance_parameter (distance_function | str)
            A distance_function or a string. If a string a lookup
            in a dict containing the distance measure functions
            built into sktime will be referred to using the str
            as a key
        """
        if type(distance) == str:
            # Look str up in dict that stores distance functions
            pass
        elif type(distance) == distance_function:
            self.distance_function = distance
        else:
            # Maybe set by default to euclidean?
            pass

    @staticmethod
    def distance_metric(distance: distance_parameter) -> distance_parameter:
        """
        Method that is used to get the distance function from a string parameter
        or determines if a custom distance has been passed

        Parameters
        ----------
        distance: distance_function | str
            A string which is then mapped to the appropriate distance function
            or a custom distnace_function

        Returns
        -------
        matric: distance_function
            The distance function to be used with the given algorithm
        """
        metric: distance_function = ""
        if type(metric) is not str:
            metric = distance
        else:
            if metric == "dtw":
                metric = dtw_distance
            # elif metric == "dtwcv":  # special case to force loocv grid search
            #     # cv in training
            #     if metric_params is not None:
            #         warnings.warn(
            #             "Warning: measure parameters have been specified for "
            #             "dtwcv. "
            #             "These will be ignored and parameter values will be "
            #             "found using LOOCV."
            #         )
            #     metric = dtw_distance
            #     self._cv_for_params = True
            #     self._param_matrix = {
            #         "metric_params": [{"w": x / 100} for x in range(0, 100)]
            #     }
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
            else:
                raise ValueError(
                    "Unrecognised distance measure: " + metric + ". Allowed "
                    "values are names from [dtw,ddtw, wdtw, wddtw, lcss,erp,"
                    "msm] or please pass a callable distance measure into"
                    "the constructor directly"
                )
        return distance
