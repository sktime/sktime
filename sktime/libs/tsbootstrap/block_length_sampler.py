from __future__ import annotations

import logging
import sys
import warnings
from typing import Callable, Optional, Union

import numpy as np
from numpy.random import Generator, default_rng
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.types import PositiveInt
from scipy.stats import pareto, weibull_min
from skbase.base import BaseObject

from tsbootstrap.utils.types import DistributionTypes, RngTypes
from tsbootstrap.utils.validate import validate_rng

if sys.version_info >= (3, 10):  # noqa: UP036
    from typing import TypeAlias
else:
    TypeAlias = type  # Fallback for earlier versions


# Configure logging for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG for more detailed logs

handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Constants for block length parameters
MIN_BLOCK_LENGTH: int = 1
DEFAULT_AVG_BLOCK_LENGTH: int = 2
MIN_AVG_BLOCK_LENGTH: int = 2

# Type Alias for Distribution Sampling Functions
DistributionSamplerFunc: TypeAlias = Callable[
    [Generator, int], Union[int, float]
]


# Registry for distribution types and their sampling functions
class DistributionRegistry:
    """
    Registry for managing supported distributions and their sampling functions.
    """

    _registry: dict[DistributionTypes, DistributionSamplerFunc] = {}

    @classmethod
    def register_distribution(
        cls,
        distribution: DistributionTypes,
        sampler_func: DistributionSamplerFunc,
    ) -> None:
        """
        Register a new distribution and its sampling function.

        Parameters
        ----------
        distribution : DistributionTypes
            The distribution type to register.
        sampler_func : DistributionSamplerFunc
            The sampling function corresponding to the distribution.

        Raises
        ------
        ValueError
            If the distribution is already registered.
        """
        if distribution in cls._registry:
            raise ValueError(
                f"Distribution '{distribution.value}' is already registered."
            )
        cls._registry[distribution] = sampler_func
        logger.debug(f"Registered distribution '{distribution.value}'.")

    @classmethod
    def get_sampler(
        cls, distribution: DistributionTypes
    ) -> DistributionSamplerFunc:
        """
        Retrieve the sampling function for a given distribution.

        Parameters
        ----------
        distribution : DistributionTypes
            The distribution type for which to retrieve the sampling function.

        Returns
        -------
        DistributionSamplerFunc
            The sampling function associated with the distribution.

        Raises
        ------
        ValueError
            If the distribution is not registered.
        """
        try:
            sampler = cls._registry[distribution]
        except KeyError:
            raise ValueError(
                f"Sampler for distribution '{distribution.value}' is not registered."
            ) from None
        else:
            logger.debug(
                f"Retrieved sampler for distribution '{distribution.value}'."
            )
            return sampler


def sample_poisson(rng: Generator, avg_block_length: int) -> int:
    """Sample from a Poisson distribution."""
    return rng.poisson(lam=avg_block_length)


def sample_exponential(rng: Generator, avg_block_length: int) -> float:
    """Sample from an Exponential distribution."""
    return rng.exponential(scale=avg_block_length)


def sample_normal(rng: Generator, avg_block_length: int) -> float:
    """Sample from a Normal distribution."""
    return rng.normal(loc=avg_block_length, scale=avg_block_length / 3)


def sample_gamma(rng: Generator, avg_block_length: int) -> float:
    """Sample from a Gamma distribution."""
    shape: float = 2.0
    scale: float = avg_block_length / 2
    return rng.gamma(shape=shape, scale=scale)


def sample_beta(rng: Generator, avg_block_length: int) -> float:
    """Sample from a Beta distribution."""
    a: int = 2
    b: int = 2
    return rng.beta(a=a, b=b) * (2 * avg_block_length - 1) + 1


def sample_lognormal(rng: Generator, avg_block_length: int) -> float:
    """Sample from a Lognormal distribution."""
    mean: float = np.log(avg_block_length / 2)
    sigma: float = np.log(2)
    return rng.lognormal(mean=mean, sigma=sigma)


def sample_weibull(rng: Generator, avg_block_length: int) -> float:
    """Sample from a Weibull distribution."""
    c: float = 1.5  # Shape parameter
    return weibull_min.rvs(c=c, scale=avg_block_length, random_state=rng)


def sample_pareto(rng: Generator, avg_block_length: int) -> float:
    """Sample from a Pareto distribution."""
    b: float = 1  # Shape parameter
    return (pareto.rvs(b=b, random_state=rng) + 1) * avg_block_length


def sample_geometric(rng: Generator, avg_block_length: int) -> int:
    """Sample from a Geometric distribution."""
    p: float = 1 / avg_block_length
    return rng.geometric(p=p)


def sample_uniform(rng: Generator, avg_block_length: int) -> int:
    """Sample from a Uniform distribution."""
    return rng.integers(low=1, high=2 * avg_block_length)


def sample_none(rng: Generator, avg_block_length: int) -> int:
    """Return the average block length."""
    return avg_block_length


# Register all default distributions
DistributionRegistry.register_distribution(
    DistributionTypes.POISSON, sample_poisson
)
DistributionRegistry.register_distribution(
    DistributionTypes.EXPONENTIAL, sample_exponential
)
DistributionRegistry.register_distribution(
    DistributionTypes.NORMAL, sample_normal
)
DistributionRegistry.register_distribution(
    DistributionTypes.GAMMA, sample_gamma
)
DistributionRegistry.register_distribution(DistributionTypes.BETA, sample_beta)
DistributionRegistry.register_distribution(
    DistributionTypes.LOGNORMAL, sample_lognormal
)
DistributionRegistry.register_distribution(
    DistributionTypes.WEIBULL, sample_weibull
)
DistributionRegistry.register_distribution(
    DistributionTypes.PARETO, sample_pareto
)
DistributionRegistry.register_distribution(
    DistributionTypes.GEOMETRIC, sample_geometric
)
DistributionRegistry.register_distribution(
    DistributionTypes.UNIFORM, sample_uniform
)
DistributionRegistry.register_distribution(DistributionTypes.NONE, sample_none)


class BlockLengthSampler(BaseModel, BaseObject):
    """
    A class for sampling block lengths for the random block length bootstrap.

    This class provides functionality to sample block lengths from various
    probability distributions. It is used in time series bootstrapping
    methods where variable block lengths are required.

    Parameters
    ----------
    avg_block_length : PositiveInt, optional
        The average block length to be used for sampling. Must be greater than
        or equal to `MIN_AVG_BLOCK_LENGTH`. Default is `DEFAULT_AVG_BLOCK_LENGTH`.
    block_length_distribution : Optional[Union[str, DistributionTypes]], optional
        The probability distribution to use for sampling block lengths.
        Must be one of the values in `DistributionTypes` or a corresponding string.
        Default is `None`.
    rng : RngTypes, optional
        Random number generator for reproducibility. If not provided, a new
        default RNG will be created.

    Attributes
    ----------
    avg_block_length : PositiveInt
        The average block length used for sampling.
    block_length_distribution : Optional[DistributionTypes]
        The selected probability distribution for block length sampling.
    rng : RngTypes
        The random number generator used for sampling.

    Methods
    -------
    sample_block_length()
        Sample a block length from the selected distribution.

    Examples
    --------
    >>> from tsbootstrap.utils.block_length_sampler import BlockLengthSampler, DistributionTypes
    >>> sampler = BlockLengthSampler(avg_block_length=5, block_length_distribution=DistributionTypes.GAMMA)
    >>> block_length = sampler.sample_block_length()
    >>> print(block_length)
    6

    >>> sampler_str = BlockLengthSampler(avg_block_length=5, block_length_distribution="gamma")
    >>> block_length_str = sampler_str.sample_block_length()
    >>> print(block_length_str)
    7

    >>> sampler_none = BlockLengthSampler(avg_block_length=5)
    >>> block_length_none = sampler_none.sample_block_length()
    >>> print(block_length_none)
    5

    Notes
    -----
    The class uses Pydantic for data validation and settings management.
    It inherits from both `pydantic.BaseModel` and `skbase.base.BaseObject`.
    """

    # Model configuration using Pydantic's ConfigDict for Pydantic 2.0
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Allows extra attributes like 'test__attr'
    )

    # Define class attributes with validation
    avg_block_length: PositiveInt = Field(
        default=DEFAULT_AVG_BLOCK_LENGTH,
        description="The average block length to use for sampling.",
    )
    block_length_distribution: Optional[DistributionTypes] = Field(
        default=None,
        description="The probability distribution to use for sampling block lengths. "
        "Must be one of the values in `DistributionTypes` or a corresponding string.",
    )
    rng: RngTypes = Field(  # type: ignore
        default_factory=default_rng,
        description="Random number generator for sampling.",
    )

    # Tags for the object type
    tags: dict[str, str] = Field(
        default_factory=lambda: {"object_type": "sampler"},
        exclude=True,
    )

    @field_validator("avg_block_length", mode="before")
    @classmethod
    def validate_avg_block_length(cls, v: int) -> int:
        """
        Validate that `avg_block_length` is greater than or equal to `MIN_AVG_BLOCK_LENGTH`.

        If `v` is less than `MIN_AVG_BLOCK_LENGTH`, issue a warning and set it to `MIN_AVG_BLOCK_LENGTH`.

        Parameters
        ----------
        v : int
            The average block length to validate.

        Returns
        -------
        int
            The validated (and possibly adjusted) average block length.
        """
        if v < MIN_AVG_BLOCK_LENGTH:
            warnings.warn(
                f"avg_block_length should be an integer greater than or equal to {MIN_AVG_BLOCK_LENGTH}. "
                f"Setting to {MIN_AVG_BLOCK_LENGTH}.",
                UserWarning,
                stacklevel=3,
            )
            logger.warning(
                f"avg_block_length was {v}, which is less than {MIN_AVG_BLOCK_LENGTH}. "
                f"Setting to {MIN_AVG_BLOCK_LENGTH}."
            )
            return MIN_AVG_BLOCK_LENGTH
        logger.debug(f"avg_block_length validated: {v}")
        return v

    @field_validator("rng", mode="before")
    @classmethod
    def validate_rng_field(cls, v: Union[Generator, int, None]) -> Generator:
        """
        Validate the random number generator.

        This method ensures that the provided random number generator
        is valid and consistent with the expected type.

        Parameters
        ----------
        v : Union[Generator, int, None]
            The input random number generator to validate.

        Returns
        -------
        Generator
            The validated random number generator.

        Raises
        ------
        ValueError
            If the input is not a valid random number generator or seed.
        """
        validated_rng: Generator = validate_rng(v, allow_seed=True)
        logger.debug("Random number generator validated and initialized.")
        return validated_rng

    @field_validator("block_length_distribution", mode="before")
    @classmethod
    def validate_block_length_distribution(
        cls, v: Optional[Union[str, DistributionTypes]]
    ) -> Optional[DistributionTypes]:
        """
        Validate and normalize the block length distribution input.

        This method ensures that string inputs for `block_length_distribution`
        are converted to lowercase for consistency and then to the appropriate
        `DistributionTypes` enum value. It also handles `None` values.

        Parameters
        ----------
        v : Optional[Union[str, DistributionTypes]]
            The input block length distribution to validate.

        Returns
        -------
        Optional[DistributionTypes]
            The validated and normalized block length distribution.

        Raises
        ------
        ValueError
            If the input string is not a valid `DistributionTypes` value.
        """
        if v is None:
            logger.debug(
                "No block_length_distribution provided. Using default."
            )
            return None
        if isinstance(v, str):
            v_lower = v.lower()
            try:
                distribution = DistributionTypes(v_lower)
            except ValueError:
                raise ValueError(
                    f"Invalid distribution type: '{v}'. Supported types are: {[d.value for d in DistributionTypes]}"
                ) from None
            else:
                logger.debug(
                    f"block_length_distribution validated: {distribution.value}"
                )
                return distribution
        if isinstance(v, DistributionTypes):
            logger.debug(f"block_length_distribution validated: {v.value}")
            return v
        raise TypeError(
            "block_length_distribution must be a string corresponding to a supported distribution or None."
        )

    def __init__(self, **data):
        """
        Initialize the BlockLengthSampler, ensuring proper initialization of parent classes.

        Parameters
        ----------
        **data : dict
            Keyword arguments for initializing the class.
        """
        super().__init__(**data)
        BaseObject.__init__(self)

    def sample_block_length(self) -> int:
        """
        Sample a block length from the selected distribution.

        This method uses the configured distribution type and parameters
        to generate a random block length.

        Returns
        -------
        int
            A sampled block length. The returned value is always an integer
            and is at least `MIN_BLOCK_LENGTH`.

        Notes
        -----
        The sampled value is rounded to the nearest integer and is
        ensured to be no less than `MIN_BLOCK_LENGTH`.
        """
        if self.block_length_distribution is None:
            logger.debug(
                "No distribution selected. Returning average block length."
            )
            return self.avg_block_length

        # Retrieve the appropriate sampling function from the registry
        try:
            sampling_func: DistributionSamplerFunc = (
                DistributionRegistry.get_sampler(
                    self.block_length_distribution
                )
            )
        except ValueError:
            logger.exception(
                f"Error retrieving sampling function for distribution '{self.block_length_distribution.value}'."
            )
            raise

        # Sample from the selected distribution
        sampled_block_length: Union[int, float] = sampling_func(
            self.rng, self.avg_block_length
        )
        logger.debug(
            f"Sampled block length before rounding: {sampled_block_length}"
        )

        # Ensure the sampled length is an integer and at least MIN_BLOCK_LENGTH
        sampled_length_int: int = max(
            round(sampled_block_length), MIN_BLOCK_LENGTH
        )
        logger.debug(
            f"Sampled block length after validation: {sampled_length_int}"
        )

        return sampled_length_int
