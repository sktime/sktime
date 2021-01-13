# -*- coding: utf-8 -*-

__author__ = "Christopher Holder"
__all__ = ["Cluster"]


from typing import List

from sktime.clustering.utils import (
    check_data_parameters,
    check_multiple_data_parameters,
)
from sktime.distances.elastic_cython import ddtw_distance
from sktime.distances.elastic_cython import dtw_distance
from sktime.distances.elastic_cython import erp_distance
from sktime.distances.elastic_cython import lcss_distance
from sktime.distances.elastic_cython import msm_distance
from sktime.distances.elastic_cython import twe_distance
from sktime.distances.elastic_cython import wddtw_distance
from sktime.distances.elastic_cython import wdtw_distance
from sktime.distances.mpdist import mpdist
from sktime.clustering.types import (
    Data_Parameter,
    Metric_Parameter,
    Metric_Function_Dict,
)


class Cluster:
    """
    Base class used for clustering models. It is designed to deal with all
    the underlying sktime specific details so that new implementations can
    be used done quickly and efficiently. Additionally it seeks to make the
    use of the model as similar to the orginal design as possible

    Attributes
    ----------
    __metric_dict: Metric_Function_Dict (dict[str, Metric_Function])
        Dict that is used to get the appropriate metric function from a string
    """

    __metric_dict: Metric_Function_Dict = {
        "dtw": dtw_distance,
        "ddtw": ddtw_distance,
        "wdtw": wdtw_distance,
        "wddtw": wddtw_distance,
        "lcss": lcss_distance,
        "erp": erp_distance,
        "msm": msm_distance,
        "twe": twe_distance,
        "mpdist": mpdist,
    }

    def __init__(
        self,
        metric: Metric_Parameter = None,
    ) -> None:
        """
        Consturctor for a cluster algorithm.

        Parameters
        ----------
        model: any
            A clustering model
        metric: Metric_Parameter (Metric_Function | str)
            Metric function to be used in the clustering
            algorithm
        """
        self.cluster_metric: Metric_Parameter = metric
        self.model = None

    def fit(self, X: Data_Parameter, y: Data_Parameter = None):
        """
        Fit is a method that fits the given model
        """
        self.__check_model()
        if y is not None:
            X, y = check_multiple_data_parameters([X, y])
        else:
            X = check_data_parameters(X)
        return self.model.fit(X, y)

    def fit_predict(
        self, X: Data_Parameter, y: Data_Parameter = None, sample_weight=None
    ):
        """
        Compute cluster centers and predict cluster index for each sample
        """
        self.__check_model()
        if y is not None:
            X, y = check_multiple_data_parameters([X, y])
        else:
            X = check_data_parameters(X)
        numArgs: int = self.model.fit_predict.__code__.co_argcount
        if numArgs > 3:
            return self.model.fit_predict(X, y, sample_weight=sample_weight)
        return self.model.fit_predict(X, y)

    def fit_transform(
        self, X: Data_Parameter, y: Data_Parameter = None, sample_weight=None
    ):
        """
        Computer clustering and transform X to cluster-distance space
        """
        self.__check_model()
        if y is not None:
            X, y = check_multiple_data_parameters([X, y])
        else:
            X = check_data_parameters(X)
        return self.model.fit_transform(X, y, sample_weight=sample_weight)

    def predict(self, X: Data_Parameter, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to
        """
        self.__check_model()
        X = check_data_parameters(X)
        return self.model.predict(X, sample_weight=sample_weight)

    def score(self, y: Data_Parameter = None, sample_weight=None):
        """
        Opposite of the value of X
        """
        self.__check_model()
        y = check_data_parameters(y)
        return self.model.score(y=y, sample_weight=sample_weight)

    def model_built_in_metrics(self) -> List[str]:
        """
        Method intended to be overidden that returns a list of extra

        Returns
        -------
        arr: [str]
            Array of string values of the extra metrics
        """
        return [""]

    @property
    def cluster_metric(self):
        """
        Property for the __metric
        """
        return self.__metric

    @cluster_metric.setter
    def cluster_metric(self, metric: Metric_Parameter) -> None:
        """
        Setter method that is used to get the metric function from a string
        parameter or determines if a custom metric has been passed

        Parameters
        ----------
        metric: Metric_Function | str
            A string which is then mapped to the appropriate metric function
            or a custom distnace_function
        """
        if type(metric) is str:
            if metric in Cluster.__metric_dict:
                metric = Cluster.__metric_dict[metric]
            elif metric not in self.model_built_in_metrics():
                raise ValueError(
                    "Unrecognised distance measure: " + metric + ". Allowed "
                    "values are names from [dtw,ddtw, wdtw, wddtw, lcss,erp,"
                    "msm] or please pass a callable distance measure into"
                    "the constructor directly"
                )
        self.__metric = metric

    def __check_model(self):
        """
        Method used to check the model is set correctly
        """
        if self.model is None:
            raise ValueError(
                "No model has been supplied please set the model \
                (self.model)"
            )
