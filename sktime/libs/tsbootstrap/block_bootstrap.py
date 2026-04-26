from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import numpy as np

from tsbootstrap.base_bootstrap import BaseTimeSeriesBootstrap
from tsbootstrap.block_bootstrap_configs import (
    BartlettsBootstrapConfig,
    BaseBlockBootstrapConfig,
    BlackmanBootstrapConfig,
    BlockBootstrapConfig,
    CircularBlockBootstrapConfig,
    HammingBootstrapConfig,
    HanningBootstrapConfig,
    MovingBlockBootstrapConfig,
    NonOverlappingBlockBootstrapConfig,
    StationaryBlockBootstrapConfig,
    TukeyBootstrapConfig,
)
from tsbootstrap.block_generator import BlockGenerator
from tsbootstrap.block_length_sampler import BlockLengthSampler
from tsbootstrap.block_resampler import BlockResampler


class BlockBootstrap(BaseTimeSeriesBootstrap):
    """
    Block Bootstrap base class for time series data.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use. If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.
    """

    _tags = {"bootstrap_type": "block"}

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
    ):
        """
        Block Bootstrap class for time series data.
        """
        self.n_bootstraps = n_bootstraps
        self.block_length = block_length
        self.block_length_distribution = block_length_distribution
        self.wrap_around_flag = wrap_around_flag
        self.overlap_flag = overlap_flag
        self.combine_generation_and_sampling_flag = (
            combine_generation_and_sampling_flag
        )
        self.block_weights = block_weights
        self.tapered_weights = tapered_weights
        self.overlap_length = overlap_length
        self.min_block_length = min_block_length
        self.rng = rng

        if type(self) == BlockBootstrap:
            cgsf = combine_generation_and_sampling_flag
            self.config = BlockBootstrapConfig(
                block_length=block_length,
                block_length_distribution=block_length_distribution,
                wrap_around_flag=wrap_around_flag,
                overlap_flag=overlap_flag,
                combine_generation_and_sampling_flag=cgsf,
                block_weights=block_weights,
                tapered_weights=tapered_weights,
                overlap_length=overlap_length,
                min_block_length=min_block_length,
                rng=rng,
            )

        super().__init__(n_bootstraps=n_bootstraps, rng=rng)

        self.blocks = None
        self.block_resampler = None

    def _check_input(self, X: np.ndarray, enforce_univariate=True) -> None:
        super()._check_input(X=X, enforce_univariate=enforce_univariate)
        if self.config.block_length is not None and self.config.block_length > X.shape[0]:  # type: ignore
            raise ValueError(
                "block_length cannot be greater than the size of the input array X."
            )

    def _generate_blocks(self, X: np.ndarray):
        """Generates blocks of indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        blocks : list of arrays
            The generated blocks.

        """
        block_length_sampler = BlockLengthSampler(
            avg_block_length=(
                self.config.block_length
                if self.config.block_length is not None
                else int(np.sqrt(X.shape[0]))
            ),  # type: ignore
            block_length_distribution=self.config.block_length_distribution,
            rng=self.config.rng,
        )

        block_generator = BlockGenerator(
            block_length_sampler=block_length_sampler,
            input_length=X.shape[0],  # type: ignore
            rng=self.config.rng,
            wrap_around_flag=self.config.wrap_around_flag,
            overlap_length=self.config.overlap_length,
            min_block_length=self.config.min_block_length,
        )

        blocks = block_generator.generate_blocks(
            overlap_flag=self.config.overlap_flag
        )

        return blocks

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        """
        Generate a single bootstrap sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing the indices and data of the generated blocks.
        """
        if (
            self.config.combine_generation_and_sampling_flag
            or self.blocks is None
        ):
            blocks = self._generate_blocks(X=X)

            block_resampler = BlockResampler(
                X=X,
                blocks=blocks,
                rng=self.config.rng,
                block_weights=self.config.block_weights,
                tapered_weights=self.config.tapered_weights,
            )
        else:
            blocks = self.blocks
            block_resampler = self.block_resampler

        (
            block_indices,
            block_data,
        ) = block_resampler.resample_block_indices_and_data()  # type: ignore

        if not self.config.combine_generation_and_sampling_flag:
            self.blocks = blocks
            self.block_resampler = block_resampler

        return block_indices, block_data


class BaseBlockBootstrap(BlockBootstrap):
    """
    Base class for block bootstrapping.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample. If None, the block length is automatically set to the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use. If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.
    bootstrap_type : str, default=None
        The type of block bootstrap to use.
        Must be one of "nonoverlapping", "moving", "stationary", or "circular".
    """

    def __init__(
        self,
        bootstrap_type: str = None,
        **kwargs,
    ):
        # def __init__(
        #     self,
        #     n_bootstraps: Integral = 10,  # type: ignore
        #     block_length: Integral = None,
        #     block_length_distribution: str = None,
        #     wrap_around_flag: bool = False,
        #     overlap_flag: bool = False,
        #     combine_generation_and_sampling_flag: bool = False,
        #     block_weights=None,
        #     tapered_weights: Callable = None,
        #     overlap_length: Integral = None,
        #     min_block_length: Integral = None,
        #     rng=None,
        #     bootstrap_type: str = None,
        #     **kwargs,
        # ):
        self.bootstrap_type = bootstrap_type

        if hasattr(self, "config"):
            config = self.config
        else:
            config = BaseBlockBootstrapConfig(
                bootstrap_type=bootstrap_type,
                **kwargs,
            )
            # config = BaseBlockBootstrapConfig(
            #     n_bootstraps=n_bootstraps,
            #     block_length=block_length,
            #     block_length_distribution=block_length_distribution,
            #     wrap_around_flag=wrap_around_flag,
            #     overlap_flag=overlap_flag,
            #     combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            #     block_weights=block_weights,
            #     tapered_weights=tapered_weights,
            #     overlap_length=overlap_length,
            #     min_block_length=min_block_length,
            #     rng=rng,
            #     bootstrap_type=bootstrap_type,
            # )
            self.config = config

        super().__init__(
            # n_bootstraps=n_bootstraps,
            # block_length=block_length,
            # block_length_distribution=block_length_distribution,
            # wrap_around_flag=wrap_around_flag,
            # overlap_flag=overlap_flag,
            # combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            # block_weights=block_weights,
            # tapered_weights=tapered_weights,
            # overlap_length=overlap_length,
            # min_block_length=min_block_length,
            # rng=rng,
            **kwargs,
        )

        self.bootstrap_instance: BlockBootstrap = None

        if config.bootstrap_type:
            bcls = BLOCK_BOOTSTRAP_TYPES_DICT[config.bootstrap_type]
            # self_params = self.get_params()
            # if "bootstrap_type" in self_params:
            #    self_params.pop("bootstrap_type")
            # bcls_params = bcls.get_param_names()
            # bcls_kwargs = {k: v for k, v in self_params.items() if k in bcls_params}
            # self.bootstrap_instance = bcls(**self_params)
            self.bootstrap_instance = bcls(**kwargs)

    def _generate_samples_single_bootstrap(self, X: np.ndarray, y=None):
        """
        Generate a single bootstrap sample using either the base BlockBootstrap method or the specified bootstrap_type.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing the indices and data of the generated blocks.
        """
        if self.bootstrap_instance is None:
            # Generate samples using the base BlockBootstrap method
            (
                block_indices,
                block_data,
            ) = super()._generate_samples_single_bootstrap(X=X, y=y)
        else:
            # Generate samples using the specified bootstrap_type
            if hasattr(
                self.bootstrap_instance, "_generate_samples_single_bootstrap"
            ):
                (
                    block_indices,
                    block_data,
                ) = self.bootstrap_instance._generate_samples_single_bootstrap(
                    X=X, y=y
                )
            else:
                raise NotImplementedError(
                    f"The bootstrap class '{type(self.bootstrap_instance).__name__}' does not implement '_generate_samples_single_bootstrap' method."
                )

        return block_indices, block_data


class MovingBlockBootstrap(BlockBootstrap):
    r"""
    Moving Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Moving Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
        **kwargs,
    ):
        self.config = MovingBlockBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class StationaryBlockBootstrap(BlockBootstrap):
    r"""
    Stationary Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always "geometric", meaning that the block
    length distribution is geometrically distributed.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Stationary Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
        **kwargs,
    ):
        self.config = StationaryBlockBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class CircularBlockBootstrap(BlockBootstrap):
    r"""
    Circular Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to True, meaning that blocks can overlap.
    * `wrap_around_flag` is always set to True, meaning that the data will wrap
    around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Circular Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + \\lfloor U_i \\rfloor}

    where :math:`L` is the block length, :math:`U_i` is a uniform random variable on :math:`[0, 1]`, and :math:`\\lfloor \\cdot \\rfloor` is the floor function.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
        **kwargs,
    ):
        self.config = CircularBlockBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class NonOverlappingBlockBootstrap(BlockBootstrap):
    r"""
    Non-Overlapping Block Bootstrap class for time series data.

    This class functions similarly to the base `BlockBootstrap` class, with
    the following modifications to the default behavior:
    * `overlap_flag` is always set to False, meaning that blocks cannot overlap.
    * `wrap_around_flag` is always set to False, meaning that the data will not
    wrap around when generating blocks.
    * `block_length_distribution` is always None, meaning that the block length
    distribution is not utilized.
    * `combine_generation_and_sampling_flag` is always False, meaning that the block
    generation and resampling are performed separately.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Raises
    ------
    ValueError
        If block_length is not greater than 0.

    Notes
    -----
    The Non-Overlapping Block Bootstrap is defined as:

    .. math::
        \\hat{X}_t = \\frac{1}{L}\\sum_{i=1}^L X_{t + i}

    where :math:`L` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
        **kwargs,
    ):
        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )
        self.config = NonOverlappingBlockBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )


# Be cautious when using the default windowing functions from numpy, as they drop to 0 at the edges.This could be particularly problematic for smaller block_lengths. In the current implementation, we have clipped the min to 0.1, in block_resampler.py.


class BartlettsBootstrap(BaseBlockBootstrap):
    """Bartlett's Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Bartlett's window for tapered weights.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
        **kwargs,
    ):
        self.config = BartlettsBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class HammingBootstrap(BaseBlockBootstrap):
    r"""
    Hamming Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Hamming window for tapered weights.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Hamming window is defined as:

    .. math::
        w(n) = 0.54 - 0.46 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
        **kwargs,
    ):
        self.config = HammingBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class HanningBootstrap(BaseBlockBootstrap):
    r"""
    Hanning Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Hanning window for tapered weights.

    Parameters
    ----------
    n_bootstraps : Integral, default=10
        The number of bootstrap samples to create.
    block_length : Integral, default=None
        The length of the blocks to sample.
        If None, the block length is the square root of the number of observations.
    block_length_distribution : str, default=None
        The block length distribution function to use.
        If None, the block length distribution is not utilized.
    wrap_around_flag : bool, default=False
        Whether to wrap around the data when generating blocks.
    overlap_flag : bool, default=False
        Whether to allow blocks to overlap.
    combine_generation_and_sampling_flag : bool, default=False
        Whether to combine the block generation and sampling steps.
    block_weights : array-like of shape (n_blocks,), default=None
        The weights to use when sampling blocks.
    tapered_weights : callable, default=None
        The tapered weights to use when sampling blocks.
    overlap_length : Integral, default=None
        The length of the overlap between blocks.
    min_block_length : Integral, default=None
        The minimum length of the blocks.
    bootstrap_type : str, default=None
        The type of block bootstrap to use.
        Must be one of "nonoverlapping", "moving", "stationary", or "circular".
    rng : Integral or np.random.Generator, default=np.random.default_rng()
        The random number generator or seed used to generate the bootstrap samples.

    Notes
    -----
    The Hanning window is defined as:

    .. math::
        w(n) = 0.5 - 0.5 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        bootstrap_type: str = None,
        rng=None,
        **kwargs,
    ):
        self.config = HanningBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            bootstrap_type=bootstrap_type,
            rng=rng,
            **kwargs,
        )


class BlackmanBootstrap(BaseBlockBootstrap):
    r"""
    Blackman Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Blackman window for tapered weights.

    Notes
    -----
    The Blackman window is defined as:

    .. math::
        w(n) = 0.42 - 0.5 \\cos\\left(\\frac{2\\pi n}{N - 1}\\right) + 0.08 \\cos\\left(\\frac{4\\pi n}{N - 1}\\right)

    where :math:`N` is the block length.

    References
    ----------
    .. [^1^] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Moving_block_bootstrap
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
        **kwargs,
    ):
        self.config = BlackmanBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


class TukeyBootstrap(BaseBlockBootstrap):
    r"""
    Tukey Bootstrap class for time series data.

    This class is a specialized bootstrapping class that uses
    Tukey window for tapered weights.

    Notes
    -----
    The Tukey window is defined as:

    .. math::
        w(n) = \\begin{cases}
            0.5\\left[1 + \\cos\\left(\\frac{2\\pi n}{\\alpha(N - 1)}\\right)\\right], & \\text{if } n < \\frac{\\alpha(N - 1)}{2}\\\\
            1, & \\text{if } \\frac{\\alpha(N - 1)}{2} \\leq n \\leq (N - 1)\\left(1 - \\frac{\\alpha}{2}\\right)\\\\
            0.5\\left[1 + \\cos\\left(\\frac{2\\pi n}{\\alpha(N - 1)}\\right)\\right], & \\text{if } n > (N - 1)\\left(1 - \\frac{\\alpha}{2}\\right)
        \\end{cases}

    where :math:`N` is the block length and :math:`\\alpha` is the parameter
    controlling the shape of the window.
    """

    def __init__(
        self,
        n_bootstraps: Integral = 10,  # type: ignore
        block_length: Integral = None,
        block_length_distribution: str = None,
        wrap_around_flag: bool = False,
        overlap_flag: bool = False,
        combine_generation_and_sampling_flag: bool = False,
        block_weights=None,
        tapered_weights: Callable = None,
        overlap_length: Integral = None,
        min_block_length: Integral = None,
        rng=None,
        **kwargs,
    ):
        self.config = TukeyBootstrapConfig(
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
        )

        super().__init__(
            n_bootstraps=n_bootstraps,
            block_length=block_length,
            block_length_distribution=block_length_distribution,
            wrap_around_flag=wrap_around_flag,
            overlap_flag=overlap_flag,
            combine_generation_and_sampling_flag=combine_generation_and_sampling_flag,
            block_weights=block_weights,
            tapered_weights=tapered_weights,
            overlap_length=overlap_length,
            min_block_length=min_block_length,
            rng=rng,
            **kwargs,
        )


BLOCK_BOOTSTRAP_TYPES_DICT = {
    "nonoverlapping": NonOverlappingBlockBootstrap,
    "moving": MovingBlockBootstrap,
    "stationary": StationaryBlockBootstrap,
    "circular": CircularBlockBootstrap,
}
