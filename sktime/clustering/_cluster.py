# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["Cluster"]


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
    """

    def __init__(
        self,
        distance: distance_parameter = None,
    ) -> None:
        """
        Consturctor for a cluster algorithm.

        Parameters
        ----------
        distance: distance_parameter (distance_function | str)
            Distance function to be used in the clustering
            algorithm
        """
        self.__distance_metric: distance_parameter = distance

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

    @property
    def __distance_metric(self):
        """
        Property for the distance_metric
        """
        return self.distance_metric

    @__distance_metric.setter
    def __distance_metric(self, distance: distance_parameter) -> None:
        """
        Setter method that is used to get the distance function from a string
        parameter or determines if a custom distance has been passed

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
        if type(distance) is not str:
            distance = distance
        else:
            if distance == "dtw":
                distance = dtw_distance
            # elif distance == "dtwcv":  # special case to force loocv grid search
            #     # cv in training
            #     if distance_params is not None:
            #         warnings.warn(
            #             "Warning: measure parameters have been specified for "
            #             "dtwcv. "
            #             "These will be ignored and parameter values will be "
            #             "found using LOOCV."
            #         )
            #     distance = dtw_distance
            #     self._cv_for_params = True
            #     self._param_matrix = {
            #         "distance_params": [{"w": x / 100} for x in range(0, 100)]
            #     }
            elif distance == "ddtw":
                distance = ddtw_distance
            elif distance == "wdtw":
                distance = wdtw_distance
            elif distance == "wddtw":
                distance = wddtw_distance
            elif distance == "lcss":
                distance = lcss_distance
            elif distance == "erp":
                distance = erp_distance
            elif distance == "msm":
                distance = msm_distance
            elif distance == "twe":
                distance = twe_distance
            elif distance == "mpdist":
                distance = mpdist
            else:
                raise ValueError(
                    "Unrecognised distance measure: " + distance + ". Allowed "
                    "values are names from [dtw,ddtw, wdtw, wddtw, lcss,erp,"
                    "msm] or please pass a callable distance measure into"
                    "the constructor directly"
                )
        self.distance_metric = distance
