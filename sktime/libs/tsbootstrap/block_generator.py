import warnings
from numbers import Integral
from typing import Optional

import numpy as np
from numpy.random import Generator

from tsbootstrap.block_length_sampler import BlockLengthSampler
from tsbootstrap.utils.validate import (
    validate_block_indices,
    validate_integers,
)


class BlockGenerator:
    """
    A class that generates blocks of indices.

    Methods
    -------
    __init__
        Initialize the BlockGenerator with the given parameters.
    generate_non_overlapping_blocks()
        Generate non-overlapping block indices.
    generate_overlapping_blocks()
        Generate overlapping block indices.
    generate_blocks(overlap_flag=False)
        Generate block indices.
    """

    def __init__(
        self,
        block_length_sampler: BlockLengthSampler,
        input_length: Integral,
        wrap_around_flag: bool = False,
        rng: Optional[Generator] = None,  # noqa: UP007
        overlap_length: Optional[Integral] = None,  # noqa: UP007
        min_block_length: Optional[Integral] = None,  # noqa: UP007
    ):
        """
        Initialize the BlockGenerator with the given parameters.

        Parameters
        ----------
        block_length_sampler : BlockLengthSampler
            An instance of the BlockLengthSampler class which is used to determine the length of each block.
        input_length : Integral
            The length of the input time series.
        wrap_around_flag : bool, optional
            A flag indicating whether to allow wrap-around in the block sampling, by default False.
        rng : Generator, optional
            The random number generator.
        overlap_length : Integral, optional
            ONLY USED WHEN overlap_flag IS TRUE. The length of overlap between consecutive blocks. If None, overlap_length is set to half the length of the block.
        min_block_length : Integral, optional
            ONLY USED WHEN overlap_flag IS TRUE. The minimum length of a block. If None, min_block_length is set to the average block length from block_length_sampler.
        """
        self.input_length = input_length
        self.block_length_sampler = block_length_sampler
        self.wrap_around_flag = wrap_around_flag
        self.min_block_length = min_block_length
        self.overlap_length = overlap_length
        self.rng = rng

    @property
    def input_length(self) -> Integral:
        """The length of the input time series."""
        return self._input_length

    @property
    def block_length_sampler(self) -> BlockLengthSampler:
        """The block length sampler."""
        return self._block_length_sampler

    @property
    def wrap_around_flag(self) -> bool:
        """A flag indicating whether to allow wrap-around in the block sampling."""
        return self._wrap_around_flag

    @property
    def rng(self) -> Generator:
        """The random number generator."""
        return self._rng

    @property
    def overlap_length(self):
        """The length of overlap between consecutive blocks."""
        return self._overlap_length  # type: ignore

    @property
    def min_block_length(self):
        """The minimum length of a block."""
        return self._min_block_length

    @input_length.setter
    def input_length(self, value: Integral) -> None:
        """Set the length of the input time series."""
        self._validate_input_length(value)
        self._input_length = value

    @block_length_sampler.setter
    def block_length_sampler(self, sampler: BlockLengthSampler) -> None:
        """Set the block length sampler."""
        self._validate_block_length_sampler(sampler)
        self._block_length_sampler = sampler

    @wrap_around_flag.setter
    def wrap_around_flag(self, value: bool) -> None:
        """Set the wrap-around flag."""
        if not isinstance(value, bool):
            raise TypeError("'wrap_around_flag' must be a boolean.")
        self._wrap_around_flag = value

    @rng.setter
    def rng(self, value: Generator) -> None:
        """Set the random number generator."""
        if value is None:
            value = np.random.default_rng()
        elif not isinstance(value, Generator):
            raise TypeError(
                "'rng' must be an instance of the Generator class."
            )
        self._rng = value

    @overlap_length.setter
    def overlap_length(self, value: Integral) -> None:
        """Set the length of overlap between consecutive blocks."""
        self._overlap_length = self._validate_overlap_length(value)

    @min_block_length.setter
    def min_block_length(self, value: Integral) -> None:
        """Set the minimum length of a block."""
        self._min_block_length = self._validate_min_block_length(value)

    def _validate_input_length(self, value: Integral) -> None:
        """Private method to validate input length."""
        validate_integers(value, min_value=3)

    def _validate_block_length_sampler(
        self, sampler: BlockLengthSampler
    ) -> None:
        """Private method to validate block length sampler."""
        if not isinstance(sampler, BlockLengthSampler):
            raise TypeError(
                "The block length sampler must be an instance of the BlockLengthSampler class."
            )
        if sampler.avg_block_length > self.input_length:
            raise ValueError(
                f"'sampler.avg_block_length' must be less than or equal to 'input_length'. Got 'sampler.avg_block_length' = {sampler.avg_block_length} and 'input_length' = {self.input_length}."
            )

    def _validate_overlap_length(self, value):
        """Private method to validate overlap length.

        Parameters
        ----------
        value : Optional[Integral]
            The input overlap length.

        Returns
        -------
        Optional[Integral]
            The validated and possibly corrected overlap length.
        """
        if value is not None:
            validate_integers(value)
            if value < 1:
                warnings.warn(
                    "'overlap_length' should be >= 1. Setting it to 1.",
                    stacklevel=2,
                )
                return 1
        return value

    def _validate_min_block_length(self, value):
        """Private method to validate minimum block length, possibly correcting the value.

        Parameters
        ----------
        value : Optional[Integral]
            The input minimum block length.

        Returns
        -------
        Optional[Integral]
            The validated and possibly corrected minimum block length.
        """
        from tsbootstrap.block_length_sampler import MIN_BLOCK_LENGTH

        if value is not None:
            validate_integers(value)
            if value < MIN_BLOCK_LENGTH:
                warnings.warn(
                    f"'min_block_length' should be >= {MIN_BLOCK_LENGTH}. Setting it to {MIN_BLOCK_LENGTH}.",
                    stacklevel=2,
                )
                value = MIN_BLOCK_LENGTH

            if value > self.block_length_sampler.avg_block_length:
                warnings.warn(
                    f"'min_block_length' should be <= the 'avg_block_length' from 'block_length_sampler'. Setting it to {self.block_length_sampler.avg_block_length}.",
                    stacklevel=2,
                )
                value = self.block_length_sampler.avg_block_length

            print(
                f"min_block_length from blockgenerator, value is not none: {value}\n"
            )
        else:
            value = MIN_BLOCK_LENGTH
        print(f"min_block_length from blockgenerator: {value}\n")
        return value

    def _create_block(
        self, start_index: Integral, block_length: Integral
    ) -> np.ndarray:
        """
        Create a block of indices.

        Parameters
        ----------
        start_index : Integral
            Starting index of the block.
        block_length : Integral
            Length of the block.

        Returns
        -------
        np.ndarray
            An array representing the indices of a block in the time series.
        """
        end_index = (start_index + block_length) % self.input_length

        if start_index < end_index:
            return np.arange(start_index, end_index)
        else:
            return np.concatenate(
                (
                    np.arange(start_index, self.input_length),
                    np.arange(0, end_index),
                )
            )

    def _calculate_start_index(self) -> Integral:
        """
        Calculate the starting index of a block.

        Returns
        -------
        Integral
            The starting index of the block.
        """
        if self.wrap_around_flag:
            return self.rng.integers(self.input_length)  # type: ignore
        else:
            return 0  # type: ignore

    def _calculate_overlap_length(
        self, sampled_block_length: Integral
    ) -> Integral:
        """
        Calculate the overlap length for a block.

        Parameters
        ----------
        sampled_block_length : Integral
            The length of the sampled block.

        Returns
        -------
        Integral
            The calculated overlap length.
        """
        if self.overlap_length is None:
            return sampled_block_length // 2  # type: ignore
        else:
            return min(max(self.overlap_length, 1), sampled_block_length - 1)  # type: ignore

    def _get_total_length_covered(
        self, block_length: Integral, overlap_length: Integral
    ) -> Integral:
        """
        Get the total length covered in the time series considering the current block length and overlap length.

        Parameters
        ----------
        block_length : Integral
            The current block length.
        overlap_length : Integral
            The overlap length between the current and next block.

        Returns
        -------
        Integral
            The total length covered so far.
        """
        # if not self.wrap_around_flag:
        return block_length - overlap_length
        # return 0

    def _get_next_block_length(
        self, sampled_block_length: Integral, total_length_covered: Integral
    ) -> Integral:
        """
        Get the next block length after considering wrap-around and total length covered.

        Parameters
        ----------
        sampled_block_length : Integral
            The sampled block length from the block length sampler.
        total_length_covered : Integral
            The total length covered so far.

        Returns
        -------
        Integral
            The adjusted block length.
        """
        if not self.wrap_around_flag:
            return min(
                sampled_block_length, self.input_length - total_length_covered
            )
        return sampled_block_length

    def _calculate_next_start_index(
        self,
        start_index: Integral,
        block_length: Integral,
        overlap_length: Integral,
    ) -> Integral:
        """
        Calculate the next start index for generating the subsequent block.

        Parameters
        ----------
        start_index : Integral
            The start index of the current block.
        block_length : Integral
            The length of the current block.
        overlap_length : Integral
            The overlap length between the current and next block.

        Returns
        -------
        Integral
            The start index for the next block.
        """
        next_start_index = start_index + block_length - overlap_length
        next_start_index = next_start_index % self.input_length
        return next_start_index

    def generate_non_overlapping_blocks(self):
        """
        Generate non-overlapping block indices in the time series.

        Returns
        -------
        List[np.ndarray]
            List of numpy arrays containing the indices for each non-overlapping block.

        Example
        -------
        >>> block_generator = BlockGenerator(input_length=100, block_length_sampler=UniformBlockLengthSampler())
        >>> non_overlapping_blocks = block_generator.generate_non_overlapping_blocks()
        >>> len(non_overlapping_blocks)
        10
        """
        """
        Generate non-overlapping block indices.

        Returns
        -------
        list of numpy.ndarray
            A list of non-overlapping block indices.

        Raises
        ------
        ValueError
            If the block length sampler is not set.
        """
        block_indices = []
        start_index = self._calculate_start_index()
        total_length = 0

        while total_length < self.input_length:
            sampled_block_length = (
                self.block_length_sampler.sample_block_length()
            )
            block_length = self._get_next_block_length(
                sampled_block_length, total_length
            )
            block = self._create_block(start_index, block_length)
            block_indices.append(block)
            total_length += block_length
            start_index = self._calculate_next_start_index(
                start_index, block_length, overlap_length=0
            )

        validate_block_indices(block_indices, self.input_length)
        return block_indices

    def generate_overlapping_blocks(self):
        """
        Generate overlapping block indices in the time series.

        Returns
        -------
        List[np.ndarray]
            List of numpy arrays containing the indices for each overlapping block.

        Example
        -------
        >>> block_generator = BlockGenerator(input_length=100, block_length_sampler=UniformBlockLengthSampler(), overlap_length=5)
        >>> overlapping_blocks = block_generator.generate_overlapping_blocks()
        >>> len(overlapping_blocks)
        15
        """
        r"""
        Generate block indices for overlapping blocks in a time series.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.

        Notes
        -----
        The block indices are generated as follows:

        1. A starting index is sampled from a uniform distribution over the time series.
        2. A block length is sampled from the block length sampler.
        3. An overlap length is calculated from the block length.
        4. A block is created from the starting index and block length.
        5. The starting index is updated to the next starting index.
        6. Steps 2-5 are repeated until the total length covered is equal to the length of the time series.

        The block length sampler is used to sample the block length. The overlap length is calculated from the block length.
        The starting index is updated to the next starting index by adding the block length and subtracting the overlap length.
        The starting index is then wrapped around if the wrap-around flag is set to True.
        """
        block_indices = []
        start_index = self._calculate_start_index()
        total_length_covered = 0
        start_indices = []

        while total_length_covered < self.input_length:
            start_indices.append(start_index)
            sampled_block_length = (
                self.block_length_sampler.sample_block_length()
            )
            print(f"sampled_block_length: {sampled_block_length}\n")
            block_length = self._get_next_block_length(
                sampled_block_length, total_length_covered
            )
            if block_length < self.min_block_length:
                # print(f"block_length: {block_length}, min_block_length: {self.min_block_length}\n")
                # block_length = self.min_block_length
                break
            overlap_length = self._calculate_overlap_length(block_length)

            block = self._create_block(start_index, block_length)
            block_indices.append(block)

            total_length_covered += self._get_total_length_covered(
                len(block), overlap_length
            )
            start_index = self._calculate_next_start_index(
                start_index, block_length, overlap_length
            )

            if start_index in start_indices:
                break
            print(
                f"input_length: {self.input_length}, block_length: {block_length}, overlap_length: {overlap_length}, total_length_covered: {total_length_covered}, start_index: {start_index}, block: {block}\n"
            )

        validate_block_indices(block_indices, self.input_length)
        return block_indices

    def generate_blocks(self, overlap_flag: bool = False):
        """
        Generate block indices.

        This method is a general entry point to generate either overlapping or non-overlapping blocks based on the given flag.

        Parameters
        ----------
        overlap_flag : bool, optional
            A flag indicating whether to generate overlapping blocks, by default False.

        Returns
        -------
        List[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.
        """
        if overlap_flag:
            return self.generate_overlapping_blocks()
        else:
            return self.generate_non_overlapping_blocks()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_length={self.input_length}, block_length_sampler={self.block_length_sampler}, overlap_length={self.overlap_length}, wrap_around_flag={self.wrap_around_flag}, rng={self.rng})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with input length {self.input_length}, block length sampler {self.block_length_sampler}, overlap length {self.overlap_length}, wrap around flag {self.wrap_around_flag}, and random number generator {self.rng}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockGenerator):
            return (
                self.input_length == other.input_length
                and self.block_length_sampler == other.block_length_sampler
                and self.overlap_length == other.overlap_length
                and self.wrap_around_flag == other.wrap_around_flag
                and self.rng == other.rng
            )
        return False
