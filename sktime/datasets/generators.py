# -*- coding: utf-8 -*-
"""
Generators for time series simulation
"""
import math
import random
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.random import RandomState, normal
from pandas import Series
from statsmodels.tsa.arima_process import arma_generate_sample

__all__ = ["ArmaGenerator", "LinearGenerator", "NoiseGenerator", "ShapeletGenerator"]


class Generator(ABC):
    """
    Abstrct class for generators.
    """

    @abstractmethod
    def sample(self, n_sample: int) -> Series:
        """
        Sample from the generator.
        """
        pass


class NoiseGenerator(Generator):
    """
    Generator for noise from the standard normal.
    Parameters
    ----------
    random_state : int or RandomState instance, default=None
        Random state generator controls generations of random
        sequences for reproducible results over multiple calls.
    Examples
    --------
    >>> from sktime.datasets.generators import ArmaGenerator
    >>> import numpy as np
    >>> arma_generator = ArmaGenerator(ar=np.array([0.9]),
    ...                                ma=np.array([0.7, 0.3]),
    ...                                random_state=42)
    >>> sample = arma_generator.sample()
    """

    def __init__(self, random_state: Union[int, RandomState] = None) -> None:
        # use random state generation
        if isinstance(random_state, RandomState):
            self.random_state = random_state
        else:
            self.random_state = RandomState(random_state)

    def sample(self, n_sample: int = 100) -> Series:
        """
        Generate a sample from the generator.
        Parameters
        ----------
        n_sample : int, default=100
            Number of sample to generate.
        Returns
        -------
        Series
            A sample from a standard normal random process.
        """

        return pd.Series(self.random_state.normal(size=n_sample))


class ArmaGenerator(Generator):
    """
    Generator for ARMA processes for specified lag-polynomials.
    Parameters
    ----------
    ar : ndarray
        Coefficient for autoregressive lag polynomial.
        Must be entered without sign negation see example.
    ma : ndarray
        Coefficient for moving-average lag polynomial.
    random_state : int or RandomState instance, default=None
        Random state generator controls generations of random
        sequences for reproducible results over multiple calls.
    Examples
    --------
    ARMA(1,2) with AR coefficient 0.9 and MA coefficients 0.7 and 0.3
    >>> from sktime.datasets.generators import ArmaGenerator
    >>> import numpy as np
    >>> arma_generator = ArmaGenerator(ar=np.array([0.9]),
    ...                                ma=np.array([0.7, 0.3]),
    ...                                random_state=42)
    >>> sample = arma_generator.sample()
    """

    def __init__(
        self,
        ar: ndarray = None,
        ma: ndarray = None,
        random_state: Union[int, RandomState] = None,
    ) -> None:

        # if either set of coef is missing, set to 1
        # set to coef sets to ndarrays (if not)
        # convert from coef to ar/ma polynomials
        if ar is None:
            ar = np.array([1.0])
        else:
            self.arparams = np.r_[1, -np.asarray(ar)]
        if ma is None:
            ma = np.array([1.0])
        else:
            self.maparams = np.r_[1, np.asarray(ma)]

        # use random state generation
        if isinstance(random_state, RandomState):
            self.random_state = random_state
        else:
            self.random_state = RandomState(random_state)

    def sample(self, n_sample: int = 100, burnin: int = 0) -> Series:
        """
        Generate a sample from the generator.
        Parameters
        ----------
        n_sample : int, default=100
            Number of sample to generate.
        burnin : int, default=0
            Number of sample at the beginning to drop.
            Used to reduce the dependence on initial values.
        Returns
        -------
        Series
            Sample from an ARMA process.
        """
        return pd.Series(
            arma_generate_sample(
                self.arparams,
                self.maparams,
                n_sample,
                distrvs=self.random_state.normal,
                burnin=burnin,
            )
        )


class LinearGenerator(Generator):
    """
    Generator for linear process with noise.
    Parameters
    ----------
    slope : float
        Slope of linear process.
    intercept : float
        Intercept of linear process.
    noise_generator : Generator, default=None
        A noise generator for the linear signal.
        Generator should be derived from
        `sktime.datasets.generators.Generator`.
        No noise will be produced when `None` is specified.
    Examples
    --------
    Linear process with ARMA(1,2) noise.
    ARMA(1,2) with AR coefficient 0.9 and MA coefficients 0.7 and 0.3.
    >>> from sktime.datasets.generators import ArmaGenerator, LinearGenerator
    >>> import numpy as np
    >>> # create a arma generator for noise on the linear signal
    >>> arma_generator = ArmaGenerator(ar=np.array([0.9]),
    ...                                ma=np.array([0.7, 0.3]),
    ...                                random_state=42)
    >>> # create a linear generator with ARMA noise
    >>> linear_generator = LinearGenerator(0.01, 0,
    ...                                    arma_generator)
    >>> # generator sample from linear process with arma noise
    >>> sample = linear_generator.sample()
    """

    def __init__(
        self, slope: float, intercept: float, noise_generator: Generator = None
    ) -> None:
        self.slope = slope
        self.intercept = intercept
        self.noise_generator = noise_generator

    def sample(self, n_sample: int = 100) -> Series:
        """
        Generate a sample from the generator.
        Parameters
        ----------
        n_sample : int, default=100
            Number of sample to generate.
        Returns
        -------
        Series
            A sample from a linear process.
        """
        signal = np.arange(n_sample) * self.slope + self.intercept
        if self.noise_generator is not None:
            noise = self.noise_generator.sample(n_sample)
            signal = signal + noise

        return pd.Series(signal)


class ShapeletGenerator:
    """
    Generator of shapelets for Shapelet Based classifiers.
    Parameters
    ----------
    series_length : int
        Length of series to be generated

    n_shapelets : int
        Number of shapelets

    shapelet_length : int
        Length of each shapelet

    shapes : List[Shape]
        List of shapes to generate values for.
        Unless specified, shapes will be randomly generated.

    """

    DEFAULT_NUM_SHAPELETS = 5
    DEFAULT_SERIES_LENGTH = 500
    DEFAULT_SHAPELET_LENGTH = 29

    def __init__(
        self,
        series_length=DEFAULT_SERIES_LENGTH,
        n_shapelets=DEFAULT_NUM_SHAPELETS,
        shapelet_length=DEFAULT_SHAPELET_LENGTH,
        shapes=None,
    ):
        self.series_length = series_length
        self.n_shapelets = n_shapelets
        self.shapelet_length = shapelet_length

        if shapes is None:
            self.generate_random_shapes()
        else:
            self.shapes = shapes

        self.t = 0

    def generate_random_shapes(self):
        self.shapes = []
        for _ in range(self.n_shapelets):
            new_shape = self.Shape(length=self.shapelet_length)
            self.shapes.append(new_shape)

    def set_shapelet_length(self, new_length):
        self.shapelet_length = new_length
        for shape in self.shapes:
            shape.set_length(new_length)

    def generate_series(self, n):
        self.reset()
        for shape in self.shapes:
            shape.randomise_location(self.series_length)
        d = np.arange(n, dtype=float)
        for i in range(n):
            d[i] = self.generate_next()
        return pd.Series(d)

    def generate_next(self) -> float:
        value = self.generate(self.t)
        self.t += 1
        return value

    def generate(self, x) -> float:
        value = normal()
        for shape in self.shapes:
            value += shape.generate(int(x))
        return value

    def set_series_length(self, new_len):
        self.series_length = new_len

    def set_n_shapelets(self, new_n_shapelets):
        self.n_shapelets = new_n_shapelets

    def set_shapes(self, new_shapes):
        self.shapes = new_shapes

    def reset(self):
        self.t = 0

    class Shape:

        ShapeTypes = ["TRIANGLE", "HEADSHOULDERS", "SINE", "STEP", "SPIKE"]

        DEFAULT_BASE = -1
        DEFAULT_AMP = 4

        def __init__(
            self, length, shape_type="HEADSHOULDERS", base=DEFAULT_BASE, amp=DEFAULT_AMP
        ):
            self.type = shape_type
            self.length = length
            self.base = base
            self.amp = amp
            self.location = 0

        def generate(self, t) -> float:
            if (t < self.location) or (t > self.location + self.length - 1):
                return 0
            offset = t - self.location
            value = 0
            (lower, mid, upper) = (0, 0, 0)

            if self.type == "TRIANGLE":
                mid = self.length // 2
                if offset <= mid:
                    if offset == 0:
                        value = self.base
                    else:
                        value = ((offset / mid) * self.amp) + self.base
                else:
                    if offset >= self.length:
                        value = self.base
                    elif self.length % 2 == 1:
                        value = (
                            ((self.length - offset - 1) / mid) * self.amp
                        ) + self.base
                    else:
                        value = (((self.length - offset) / mid) * self.amp) + self.base

            elif self.type == "HEADSHOULDERS":
                lower = self.length // 3
                upper = 2 * lower
                if self.length % 3 == 2:
                    upper += 2
                if offset < lower:
                    value = (
                        (self.amp // 2)
                        * math.sin(
                            ((2 * math.pi) // ((self.length // 3 - 1) * 2)) * offset
                        )
                    ) + self.base
                elif offset >= upper:
                    value = (
                        (self.amp // 2)
                        * math.sin(
                            ((2 * math.pi) // ((self.length // 3 - 1) * 2))
                            * (offset - upper)
                        )
                    ) + self.base
                else:
                    value = (
                        self.amp
                        * math.sin(
                            ((2 * math.pi) // (((upper - lower) - 1) * 2))
                            * (offset - self.length // 3)
                        )
                    ) + self.base
                if value < self.base:
                    value = self.base

            elif self.type == "SINE":
                value = (
                    self.amp
                    * math.sin(((2 * math.pi) // (self.length - 1)) * offset)
                    // 2
                )

            elif self.type == "STEP":
                if offset < self.length // 2:
                    value = self.base
                else:
                    value = self.base + self.amp

            elif self.type == "SPIKE":
                lower = self.length // 4
                upper = 3 * lower
                if offset <= lower:
                    if offset == 0:
                        value = 0
                    else:
                        value = (-self.amp // 2) * (offset / lower)
                elif lower < offset < upper:
                    value = -self.amp // 2 + self.amp * (
                        (offset - lower) / (upper - lower - 1)
                    )
                else:
                    value = self.amp // 2 - self.amp // 2 * (
                        (offset - upper + 1) / (self.length - upper)
                    )

            return value

        def set_location(self, new_location):
            self.location = new_location

        def set_type(self, new_type):
            self.type = new_type
            if new_type == "HEADSHOULDERS":
                self.base = -self.amp // 4
            else:
                self.base = -self.amp // 2

        def set_length(self, new_length):
            self.length = new_length

        def randomise_location(self, series_length):
            start = 0
            if series_length > self.length:
                start = random.randint(0, series_length - self.length)
            self.set_location(start)

        def randomise_shape(self):
            types = self.ShapeTypes
            self.set_type(random.choice(types))
