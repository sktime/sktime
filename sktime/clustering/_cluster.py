# -*- coding: utf-8 -*-
from typing import Callable, Union

distance_function = Callable[[float, float], float]
distance_parameter = Union[distance_function, str]


class Cluster:

    distance_function: distance_function

    def __init__(self, distance: distance_parameter) -> None:
        """
        Consturctor for a cluster algorithm.

        Parameters
        ----------
        distance: distance_parameter (distance_function | str)
            Distance function to be used in the clustering
            algorithm
        """
        self.distance_function = distance

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
