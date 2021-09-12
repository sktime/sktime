"""
Extension template for time series distance between time series

How to use this:
- this is meant as a "fill in" template for easy extension
- do NOT import this file directly - it will break
- work through all the "Todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- you can add more private methods
- change docstrings for functions and the file
- once complete: use as a local library, or contribute to sktime via PR

Mandatory implements:
    distance between two time series - _distance(self, x, y)

Optional implements
    numba version of the distance between two time series - numba_distance(self x, y)
    see below under initial class example

State:
    none, this is a state-free scitype

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
from typing import Callable, Any

import numpy as np
from numba import njit

from sktime.metrics.distances.base.base import BaseDistance, NumbaSupportedDistance


# Todo: add any necessary imports here

class MyDistanceMetric(BaseDistance):
    """
    Custom time series distance. Todo: write docstring

    NOTE: There are no required constructor parameter. The parameters passed to the
    constructor should be used as arguments for the distance metric when
    .distance(x, y) or .pairwise(x, y) is called.
    Parameters
    ----------
    parama: Any
        First parameter
    paramb: Any
        Second parameter
    paramc: Any
        Third parameter
    """

    def __init__(self, parama, paramb, paramc):
        self.parama = parama
        self.paramb = paramb
        self.paramc = paramc

    # Todo: implement method to calculate the distance between one time seires and
    # another
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Method used to compute the distance between two time series.

        Behaviour: This method should take in two time series (this could be univariate
        or multivariate) and compute the distance between them returning a single float
        that is the distance

        Parameters
        ----------
        x: np.ndarray
            First time series of size n x m where n is the number of timepoints and
            m is the number of columns (features) per time point
        y: np.ndarray
            Second time series of size n x m where n is the number of timepoints and
            m is the number of columns (features) per time point

        Returns
        -------
        float
            Distance between the two time series
        """
        # Validation of the extra argument passed to constructor should be done here
        # Implementation of distance should be done here
        pass


"""
In the above example, while you haven't implemented .pairwise(x, y), it can be called
and a pairwise between time series in a matrix will be performed using the
.distance(x, y) that is implemented. This is done by calling the .distance(x, y) for 
each combination of time series between two matrices.

While this greatly simplifies initial implementation of distance, by only requiring 
the developer to work out how to perform the distance between two time series, 
it leaves much to be desired in terms of performance. To resolve this one can speed 
up the distance computation by rewriting the function using numbas @njit() decorator. 

In the example below it shows moving the logic of the distance computation outside 
of the .distance(x, y) and out of the class. The function when not attached to a class
allows for the @njit() decorator to be used without having to define types or writing
additional code (which is what you would have to do if you were using a jitclass).
In many instances creating an njit version of the distance leads to performance gains
when performing distance computation between two time series.
 
However, calling .distance(x, y) requires us to be in 'object mode' (defined in numba as
running python code). This means we can't call this function from inside another @njit
function without first coming out of nopython mode. This leads to great performance
 decreases especially if we want to do this operation multiple times
(see https://numba.pydata.org/numba-doc/latest/user/5minguide.html for more details).

As distances function fit the description of something we would probably want to call
a lot (for example for pairwise or classification or clustering) we want to be able to
run the distance without having to swap modes constantly.

This is where the NumbaSupportedDistance comes in. This interfaces requires a method
to be implemented called .numba_distance(x, y) which returns a function that does
the distance computation but in nopython mode (i.e. compiled function @njit()). This
then allows the distance to be called inside other nopython functions which is vital
to performance.

Below gives an example of a distance that implements the NumbaSupportedDistance
"""

# Todo: Implement njit version of the distance
@njit()
def distance_computation_outside_class(x, y, parama, paramb, paramc) -> float:
    """
    This function is outside the class as compiling a class to njit using jitclass is
    much more complex than simply moving the functions you want to use outside the class
    Parameters
    ----------
    x: np.ndarray
        First time series of size n x m where n is the number of timepoints and
        m is the number of columns (features) per time point
    y: np.ndarray
        Second time series of size n x m where n is the number of timepoints and
        m is the number of columns (features) per time point
    parama: Any
        Example extra parameter a
    paramb: Any
        Example extra parameter b

    Returns
    -------
    float
        distance between two time series
    """
    # implementation goes here
    pass


class MyDistanceMetric(BaseDistance, NumbaSupportedDistance):
    """
    Custom time series distance. Todo: write docstring

    NOTE: There are no required constructor parameter these should be parameters
    to pass to the distance metric when calling .distance(x, y)
    Parameters
    ----------
    parama: Any
        First parameter
    paramb: Any
        Second parameter
    paramc: Any
        Third parameter
    """

    def __init__(self, parama: Any, paramb: Any, paramc: Any):
        self.parama: Any = parama
        self.paramb: Any = paramb
        self.paramc: Any = paramc
        # Don't need to call super

    # Todo: implement method to calculate the distance between one time seires and
    # another
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Method used to compute the distance between two time series.

        Behaviour: This method should take in two time series (this could be univariate
        or multivariate) and compute the distance between them returning a single float
        that is the distance

        Parameters
        ----------
        x: np.ndarray
            First time series of size n x m where n is the number of timepoints and
            m is the number of columns (features) per time point
        y: np.ndarray
            Second time series of size n x m where n is the number of timepoints and
            m is the number of columns (features) per time point

        Returns
        -------
        float
            Distance between the two time series
        """
        # Validation of the extra argument passed to constructor should be done here
        # Implementation of distance should be done here
        pass

    # Todo: Implement numba version of distance metric
    def numba_distance(self, x, y) -> Callable[[np.ndarray, np.ndarray], float]:
        """
        This method should return a numba compiled version of the distance to be used
        inside .pairwise(x, y) or other nopython functions.

        Parameters
        ----------
        x: np.ndarray
            First time series of size n x m where n is the number of timepoints and
            m is the number of columns (features) per time point
        y: np.ndarray
            Second time series of size n x m where n is the number of timepoints and
            m is the number of columns (features) per time point

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], float]
            Nopython Callable that must have two parameters representing the first and
            second time series. It should then return a float as shown below. NOTE:
            while it must have an x and y, additional parameter can be provided but they
            MUST be optional
        """

        # Validation of the extra argument passed to constructor should be done here
        # Ideally validation of the x and y is done but potentially not all validation
        # can be done here if you don't know ahead of time what x and y look like in
        # every instance. If that is the case make sure some is done inside the below
        # numba_compiled_version
        @njit()
        def numba_compiled_version(
                x: np.ndarray,
                y: np.ndarray,
                parama: Any = self.parama,
                paramb: Any = self.paramb,
                paramc: Any = self.paramc
        ) -> float:
            """
            In theory numba_compiled_version should only be numba_compiled_version(x, y)
            however, we can use a hacky solution to compile the function with default
            values to be used during the computation. To do so we have the
            extra parameters default to the values we would would to pass on.
            This allows us to use kwargs in njit effectively as pass **kwargs to an njit
            function does not work. To reiterate parameter x and y are required but
            any other parameters MUST BE optional.
            """
            return distance_computation_outside_class(
                x,
                y,
                parama,
                paramb,
                paramc
            )

        # The compiled numba function is returned
        return numba_compiled_version
