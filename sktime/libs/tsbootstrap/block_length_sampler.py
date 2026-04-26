MIN_BLOCK_LENGTH = 1
DEFAULT_AVG_BLOCK_LENGTH = 2
MIN_AVG_BLOCK_LENGTH = 2
DISTRIBUTION_METHODS = {
    "none": lambda rng, avg_block_length: avg_block_length,
    "poisson": lambda rng, avg_block_length: rng.poisson(avg_block_length),
    "exponential": lambda rng, avg_block_length: rng.exponential(
        avg_block_length
    ),
    "normal": lambda rng, avg_block_length: rng.normal(
        loc=avg_block_length, scale=avg_block_length / 3
    ),
    "gamma": lambda rng, avg_block_length: rng.gamma(
        shape=2.0, scale=avg_block_length / 2
    ),
    "beta": lambda rng, avg_block_length: rng.beta(a=2, b=2)
    * (2 * avg_block_length - 1)
    + 1,
    "lognormal": lambda rng, avg_block_length: rng.lognormal(
        mean=np.log(avg_block_length / 2), sigma=np.log(2)
    ),
    "weibull": lambda rng, avg_block_length: weibull_min.rvs(
        1.5, scale=avg_block_length, rng=rng
    ),
    "pareto": lambda rng, avg_block_length: (pareto.rvs(1, rng=rng) + 1)
    * avg_block_length,
    "geometric": lambda rng, avg_block_length: rng.geometric(
        p=1 / avg_block_length
    ),
    "uniform": lambda rng, avg_block_length: rng.randint(
        low=1, high=2 * avg_block_length
    ),
}

import warnings
from numbers import Integral

import numpy as np
from numpy.random import Generator
from scipy.stats import pareto, weibull_min
from skbase.base import BaseObject

from tsbootstrap.utils.types import RngTypes
from tsbootstrap.utils.validate import validate_integers, validate_rng


class BlockLengthSampler(BaseObject):
    """
    A class for sampling block lengths for the random block length bootstrap.

    Methods
    -------
    sample_block_length()
        Sample a block length from the selected distribution.
    """

    _tags = {"object_type": "sampler"}

    def __init__(self, avg_block_length: Integral = DEFAULT_AVG_BLOCK_LENGTH, block_length_distribution: str = None, rng: RngTypes = None):  # type: ignore
        """
        Initialize the BlockLengthSampler with the selected distribution and average block length.

        Parameters
        ----------
        avg_block_length : int
            The average block length to be used for sampling. Default is 2.
        block_length_distribution : str, optional
            The block length distribution function to use, represented by its name as a string. Default is None.
        rng : int, optional
            Random seed for reproducibility, by default None. If None, the global random state is used.

        Example
        -------
        >>> block_length_sampler = BlockLengthSampler(avg_block_length=3, block_length_distribution="uniform")
        >>> block_length_sampler.sample_block_length()
        5
        """
        self.block_length_distribution = block_length_distribution
        self.avg_block_length = avg_block_length
        self.rng = rng

        super().__init__()

    @property
    def block_length_distribution(self) -> str:
        """Getter for block_length_distribution."""
        return self._block_length_distribution

    @block_length_distribution.setter
    def block_length_distribution(self, value) -> None:
        """
        Setter for block_length_distribution. Performs validation on assignment.

        Parameters
        ----------
        value : str
            The block length distribution function to use.
        """
        if value is None:
            value = "none"
        if not isinstance(value, str):
            raise TypeError("block_length_distribution must be a string")
        value = value.lower()
        if value not in DISTRIBUTION_METHODS:
            raise ValueError(f"Unknown block_length_distribution '{value}'")
        self._block_length_distribution = value

    @property
    def avg_block_length(self):
        """Getter for avg_block_length."""
        return self._avg_block_length

    @avg_block_length.setter
    def avg_block_length(self, value: Integral) -> None:
        """
        Setter for avg_block_length. Performs validation on assignment.

        Parameters
        ----------
        value : int
            The average block length to be used for sampling.
        """
        self._avg_block_length = self._validate_avg_block_length(value)

    def _validate_avg_block_length(self, value: Integral) -> Integral:
        """
        Validates the average block length.

        Parameters
        ----------
        value : int
            The average block length to be validated.

        Returns
        -------
        int
            The validated average block length.

        Raises
        ------
        ValueError
            If the average block length is less than MIN_AVG_BLOCK_LENGTH.
        """
        validate_integers(value)
        if value < MIN_AVG_BLOCK_LENGTH:
            warnings.warn(
                f"avg_block_length should be an integer greater than or equal to {MIN_AVG_BLOCK_LENGTH}. Setting to {MIN_AVG_BLOCK_LENGTH}.",
                stacklevel=3,
            )
            return MIN_AVG_BLOCK_LENGTH
        return value

    @property
    def rng(self) -> Generator:
        """Getter for rng."""
        return self._rng

    @rng.setter
    def rng(self, value: RngTypes) -> None:
        """
        Setter for rng. Performs validation on assignment.

        Parameters
        ----------
        value : int or np.random.Generator
            The random seed for reproducibility. If None, the global random state is used.
        """
        self._rng = validate_rng(value, allow_seed=True)

    def sample_block_length(self) -> int:
        """
        Sample a block length from the selected distribution.

        Returns
        -------
        int
            A sampled block length.
        """
        sampled_block_length = DISTRIBUTION_METHODS[
            self.block_length_distribution
        ](self.rng, self.avg_block_length)
        return max(round(sampled_block_length), MIN_BLOCK_LENGTH)
