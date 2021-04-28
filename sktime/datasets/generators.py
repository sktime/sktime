# -*- coding: utf-8 -*-
"""
Generators for time series simulation
"""

__author__ = ["Stuart Miller"]

from abc import abstractmethod
from typing import Union

import numpy as np
from numpy.random import RandomState
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample

from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.base import BaseEstimator


__all__ = ["ArmaGenerator", "LinearGenerator", "NoiseGenerator"]


class Generator(BaseEstimator):
    """
    A base class for generators.

    Notes
    -----
    All generator parameters should be specified as keyword arguments in
    their `__init__`.
    All generators should implement the unified interface given below.

    """

    @abstractmethod
    def sample(self, n_sample, n_instance) -> Union[Series, DataFrame]:
        """
        Sample from the generator.

        Parameters
        ----------
        n_sample : int
            The length of the series to create.
        n_instance : int
            The number of series to create.

        Returns
        -------
        sample : Series or DataFrame
            Series should be returned for 1d time series, DataFrame with dimensions
            `(n_instances, n_samples)` should be returned for 2d time series
        """
        raise NotImplementedError


class NoiseGenerator(Generator):
    """
    Generator for noise from the standard normal.

    Parameters
    ----------
    random_state : int or RandomState instance, default=None
        Random state generator controls generations of random
        sequences for reproducible results over multiple calls.

    Notes
    -----
    When multiple series are generated, the resulting DataFrame will have dimentsions
    `(n_instances, n_samples)`.
    This can be converted to nested format with `from_2d_array_to_nested` from
    `sktime.utils.data_processing`.

    Examples
    --------
    >>> from sktime.datasets.generators import NoiseGenerator
    >>> import numpy as np
    >>> noise_generator = NoiseGenerator(random_state=42)
    >>> # generate 100 samples for 1d series
    >>> sample = noise_generator.sample(100, 1)
    >>>
    >>> # generate 100 samples for 2d series
    >>> sample = noise_generator.sample(100, 2)
    >>> # convert to nested format
    >>> from sktime.utils.data_processing import from_2d_array_to_nested
    >>> nested_sample = from_2d_array_to_nested(sample)
    """

    def __init__(self, random_state: Union[int, RandomState] = None) -> None:
        # use random state generation
        if isinstance(random_state, RandomState):
            self.random_state = random_state
        else:
            self.random_state = RandomState(random_state)

    def sample(self, n_sample: int, n_instance: int) -> Union[Series, DataFrame]:
        """
        Generate a sample from the generator.

        Parameters
        ----------
        n_sample : int
            Number of sample to generate.
        n_instance : int
            Number of series to generate.

        Returns
        -------
        sample : Series or DataFrame
            A sample from a standard normal random process.
        """

        if n_sample < 1:
            raise ValueError(
                "Value of n_sample must be greater than 0. "
                f"Got {n_sample} for value of n_sample."
            )

        if n_instance == 1:
            return pd.Series(self.random_state.normal(size=n_sample))
        elif n_instance > 1:
            return pd.DataFrame(self.random_state.normal(size=(n_instance, n_sample)))
        else:
            raise ValueError(
                "Value of n_instance must be greater than 1. "
                f"Got {n_instance} for value of n_instance."
            )


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
    burnin : int, default=0
        Number of sample at the beginning to drop.
        Used to reduce the dependence on initial values.
    random_state : int or RandomState instance, default=None
        Random state generator controls generations of random
        sequences for reproducible results over multiple calls.

    Notes
    -----
    When multiple series are generated, the resulting DataFrame will have dimentsions
    `(n_instances, n_samples)`.
    This can be converted to nested format with `from_2d_array_to_nested` from
    `sktime.utils.data_processing`.

    Examples
    --------
    ARMA(1,2) with AR coefficient 0.9 and MA coefficients 0.7 and 0.3

    >>> from sktime.datasets.generators import ArmaGenerator
    >>> import numpy as np
    >>> arma_generator = ArmaGenerator(ar=np.array([0.9]),
    ...                                ma=np.array([0.7, 0.3]),
    ...                                random_state=42)
    >>> # generate 100 samples for 1d series
    >>> sample = arma_generator.sample(100, 1)
    >>>
    >>> # generate 100 samples for 2d series
    >>> sample = arma_generator.sample(100, 2)
    >>> # convert to nested format
    >>> from sktime.utils.data_processing import from_2d_array_to_nested
    >>> nested_sample = from_2d_array_to_nested(sample)
    """

    def __init__(
        self,
        ar: ndarray = None,
        ma: ndarray = None,
        burnin: int = 0,
        random_state: Union[int, RandomState] = None,
    ) -> None:

        # set burnin
        self.burnin = burnin

        # if either set of coef is missing, set to 1
        # set to coef sets to ndarrays (if not)
        # convert from coef to ar/ma polynomials
        if ar is None:
            self.ar = np.array([1.0])
        else:
            self.ar = np.r_[1, -np.asarray(ar)]
        if ma is None:
            self.ma = np.array([1.0])
        else:
            self.ma = np.r_[1, np.asarray(ma)]

        # use random state generation
        if isinstance(random_state, RandomState):
            self.random_state = random_state
        else:
            self.random_state = RandomState(random_state)

    def sample(self, n_sample: int, n_instance: int) -> Union[Series, DataFrame]:
        """
        Generate a sample from the generator.

        Parameters
        ----------
        n_sample : int
            Number of sample to generate.
        n_instance : int
            Number of series to generate

        Returns
        -------
        sample : Series or DataFrame
            Sample from an ARMA process.
        """

        if n_sample < 1:
            raise ValueError(
                "Value of n_sample must be greater than 0. "
                f"Got {n_sample} for value of n_sample."
            )

        # generate sample
        samp = arma_generate_sample(
            self.ar,
            self.ma,
            (n_instance, n_sample),
            distrvs=self.random_state.normal,
            burnin=self.burnin,
        )

        if n_instance == 1:
            return pd.Series(samp.ravel())
        elif n_instance > 1:
            return pd.DataFrame(samp)
        else:
            raise ValueError(
                "Value of n_instance must be greater than 1. "
                f"Got {n_instance} for value n_instance."
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

    Notes
    -----
    When multiple series are generated, the resulting DataFrame will have dimentsions
    `(n_instances, n_samples)`.
    This can be converted to nested format with `from_2d_array_to_nested` from
    `sktime.utils.data_processing`.

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
    >>> # generate 100 samples for 1d series
    >>> sample = linear_generator.sample(100, 1)
    >>>
    >>> # generate 100 samples for 2d series
    >>> sample = linear_generator.sample(100, 2)
    >>> # convert to nested format
    >>> from sktime.utils.data_processing import from_2d_array_to_nested
    >>> nested_sample = from_2d_array_to_nested(sample)
    """

    def __init__(
        self, slope: float, intercept: float, noise_generator: Generator = None
    ) -> None:

        self.slope = slope
        self.intercept = intercept
        self.noise_generator = noise_generator

    def sample(self, n_sample: int, n_instance: int) -> Series:
        """
        Generate a sample from the generator.

        Parameters
        ----------
        n_sample : int
            Number of sample to generate.
        n_instance : int
            Number of series to generate.

        Returns
        -------
        Series
            A sample from a linear process.
        """

        if n_sample < 1:
            raise ValueError(
                "Value of n_sample must be greater than 0. "
                f"Got {n_sample} for value of n_sample."
            )

        if n_instance == 1:
            signal = (
                np.linspace(0, n_sample - 1, num=n_sample) * self.slope + self.intercept
            )
        elif n_instance > 1:
            signal = (
                np.linspace(
                    (0,) * n_instance, (n_sample - 1,) * n_instance, num=n_sample
                ).T
                * self.slope
                + self.intercept
            )
        else:
            raise ValueError(
                "Value of n_instance must be greater than 1. "
                f"Got {n_instance} for value n_instance."
            )

        if self.noise_generator is not None:
            noise = self.noise_generator.sample(n_sample, n_instance)
            signal = signal + noise

        if n_instance == 1:
            return pd.Series(signal)
        else:
            return pd.DataFrame(signal)
