# -*- coding: utf-8 -*-
"""
Generators for time series simulation
"""
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
from numpy.random import RandomState
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample

from numpy import ndarray
from pandas import Series, DataFrame

__all__ = ["ArmaGenerator", "LinearGenerator", "NoiseGenerator"]


class Generator(ABC):
    """
    Abstrct class for generators.
    """

    @abstractmethod
    def sample(self, n_sample):
        """
        Sample from the generator.
        """
        NotImplementedError


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
    >>> # generate 100 samples
    >>> sample = arma_generator.sample(100)
    """

    def __init__(self, random_state: Union[int, RandomState] = None) -> None:
        # use random state generation
        if isinstance(random_state, RandomState):
            self.random_state = random_state
        else:
            self.random_state = RandomState(random_state)

    def sample(self, n_sample: Union[int, Tuple[int]]) -> Union[Series, DataFrame]:
        """
        Generate a sample from the generator.

        Parameters
        ----------
        n_sample : int or tuple of int
            Number of sample to generate.

        Returns
        -------
        sample : Series or DataFrame
            A sample from a standard normal random process.
        """

        if isinstance(n_sample, int):
            return pd.Series(self.random_state.normal(size=n_sample))
        elif isinstance(n_sample, tuple):
            return pd.DataFrame(self.random_state.normal(size=n_sample))
        else:
            raise TypeError(
                f"Unsupported type {type(n_sample).__name__}"
                + " for n_sample, must be int or tuple of int"
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
    >>> # generate 100 samples
    >>> sample = arma_generator.sample(100)
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

    def sample(
        self, n_sample: Union[int, Tuple[int]], burnin: int = 0
    ) -> Union[Series, DataFrame]:
        """
        Generate a sample from the generator.

        Parameters
        ----------
        n_sample : int or tuple of int
            Number of sample to generate.
        burnin : int, default=0
            Number of sample at the beginning to drop.
            Used to reduce the dependence on initial values.

        Returns
        -------
        sample : Series or DataFrame
            Sample from an ARMA process.
        """
        if isinstance(n_sample, int):
            return pd.Series(
                arma_generate_sample(
                    self.arparams,
                    self.maparams,
                    n_sample,
                    distrvs=self.random_state.normal,
                    burnin=burnin,
                )
            )
        elif isinstance(n_sample, tuple):
            pd.DataFrame(
                arma_generate_sample(
                    self.arparams,
                    self.maparams,
                    n_sample,
                    distrvs=self.random_state.normal,
                    burnin=burnin,
                )
            )
        else:
            raise TypeError(
                f"Unsupported type {type(n_sample).__name__}"
                + " for n_sample, must be int or tuple of int"
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
    >>> # generate 100 samples
    >>> sample = linear_generator.sample(100)
    """

    def __init__(
        self, slope: float, intercept: float, noise_generator: Generator = None
    ) -> None:

        self.slope = slope
        self.intercept = intercept
        self.noise_generator = noise_generator

    def sample(self, n_sample) -> Series:
        """
        Generate a sample from the generator.

        Parameters
        ----------
        n_sample : int
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
