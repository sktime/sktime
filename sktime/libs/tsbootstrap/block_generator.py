from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from tsbootstrap.block_length_sampler import (
    MIN_BLOCK_LENGTH,
    BlockLengthSampler,
)
from tsbootstrap.utils.types import RngTypes
from tsbootstrap.utils.validate import validate_block_indices

# Initialize logger for this module using the module's name.
logger = logging.getLogger(__name__)


class BlockGenerator(BaseModel):
    """
    Generates blocks of indices for time series bootstrapping.

    This class is responsible for creating blocks of indices that can be used
    to sample segments from a time series. It supports both overlapping and
    non-overlapping blocks and can optionally wrap around the end of the time series.

    Parameters
    ----------
    input_length : PositiveInt
        The length of the input time series. Must be at least 3.
    block_length_sampler : BlockLengthSampler
        An instance of BlockLengthSampler to determine block lengths.
    wrap_around_flag : bool, optional
        If True, blocks can wrap around the end of the time series. Default is False.
    rng : RngTypes, optional
        Random number generator for sampling. Defaults to a new RNG instance.
    overlap_length : PositiveInt, optional
        The length of overlap between consecutive blocks. If not provided, defaults to half the average block length.
    min_block_length : PositiveInt, optional
        The minimum allowed block length. Defaults to `MIN_BLOCK_LENGTH`.

    Examples
    --------
    >>> from tsbootstrap.block_length_sampler import BlockLengthSampler, DistributionTypes
    >>> sampler = BlockLengthSampler(avg_block_length=5, block_length_distribution=DistributionTypes.GAMMA)
    >>> generator = BlockGenerator(input_length=100, block_length_sampler=sampler, wrap_around_flag=True)
    >>> blocks = generator.generate_blocks(overlap_flag=True)
    >>> print(blocks)
    [array([...]), array([...]), ...]
    """

    # Configuration for Pydantic 2 using ConfigDict
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allows custom types like Generator
        validate_assignment=True,  # Validates fields on assignment
    )

    # Define class attributes with Pydantic Fields
    block_length_sampler: BlockLengthSampler = Field(
        ..., description="Sampler for determining block lengths."
    )
    input_length: PositiveInt = Field(
        ..., ge=3, description="The length of the input time series."
    )
    wrap_around_flag: bool = Field(
        default=False,
        description="Flag to allow wrap-around in block generation.",
    )
    rng: RngTypes = Field(  # type: ignore
        default_factory=lambda: np.random.default_rng(seed=None),
        description="Random number generator.",
    )
    overlap_length: Optional[PositiveInt] = Field(
        default=0, description="Overlap length between blocks."
    )
    min_block_length: Optional[PositiveInt] = Field(
        default=MIN_BLOCK_LENGTH,
        ge=MIN_BLOCK_LENGTH,
        description="Minimum block length.",
    )

    @field_validator("block_length_sampler", mode="before")
    @classmethod
    def validate_block_length_sampler(
        cls, v: BlockLengthSampler, info: ValidationInfo
    ) -> BlockLengthSampler:
        """
        Validate the block_length_sampler to ensure its average block length does not exceed the input_length.

        Parameters
        ----------
        v : BlockLengthSampler
            The BlockLengthSampler instance to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        BlockLengthSampler
            The validated BlockLengthSampler instance.

        Raises
        ------
        ValueError
            If the sampler's average block length exceeds the input_length.
        """
        input_length: Optional[int] = info.data.get("input_length")
        if input_length is not None and v.avg_block_length > input_length:
            error_msg = (
                f"'sampler.avg_block_length' ({v.avg_block_length}) must be <= "
                f"'input_length' ({input_length})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(f"Block length sampler validated: {v}")
        return v

    @field_validator("overlap_length", mode="before")
    @classmethod
    def validate_overlap_length(
        cls, v: Optional[int], info: ValidationInfo
    ) -> int:
        """
        Validate and adjust the overlap_length parameter.

        If overlap_length is provided and is greater than or equal to input_length,
        it is adjusted to input_length - 1 with a warning.
        If overlap_length is not provided, it defaults to half of the average block length.

        Parameters
        ----------
        v : Optional[int]
            The overlap length to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        int
            The validated and possibly adjusted overlap length.

        Raises
        ------
        ValueError
            If input_length or block_length_sampler is not provided.
        """
        input_length: Optional[int] = info.data.get("input_length")
        block_length_sampler: Optional[BlockLengthSampler] = info.data.get(
            "block_length_sampler"
        )

        if input_length is None or block_length_sampler is None:
            error_msg = (
                "'input_length' and 'block_length_sampler' must be provided."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if v is None:
            # Default to half of the average block length if overlap_length is not provided
            default_overlap: int = block_length_sampler.avg_block_length // 2
            logger.debug(
                f"No 'overlap_length' provided. Defaulting to {default_overlap}."
            )
            return default_overlap
        if v >= input_length:
            # Adjust overlap_length to input_length - 1 if it exceeds or equals input_length
            adjusted_overlap = input_length - 1
            warnings.warn(
                f"'overlap_length' ({v}) should be < 'input_length' ({input_length}). "
                f"Setting to {adjusted_overlap}.",
                stacklevel=2,
            )
            logger.warning(
                f"'overlap_length' ({v}) exceeds or equals 'input_length' ({input_length}). "
                f"Adjusted to {adjusted_overlap}."
            )
            return adjusted_overlap
        logger.debug(f"'overlap_length' validated: {v}")
        return v

    @field_validator("min_block_length", mode="before")
    @classmethod
    def validate_min_block_length(
        cls, v: Optional[int], info: ValidationInfo
    ) -> int:
        """
        Validate and adjust the min_block_length parameter.

        If min_block_length is provided, it must be between MIN_BLOCK_LENGTH and
        the sampler's average block length. Otherwise, it defaults to MIN_BLOCK_LENGTH.

        Parameters
        ----------
        v : Optional[int]
            The minimum block length to validate.
        info : ValidationInfo
            Provides context about the validation, including other fields.

        Returns
        -------
        int
            The validated and possibly adjusted minimum block length.

        Raises
        ------
        ValueError
            If block_length_sampler is not provided.
        """
        block_length_sampler: Optional[BlockLengthSampler] = info.data.get(
            "block_length_sampler"
        )

        if block_length_sampler is None:
            error_msg = "'block_length_sampler' must be provided."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if v is None:
            # Default to MIN_BLOCK_LENGTH if min_block_length is not provided
            logger.debug(
                f"No 'min_block_length' provided. Defaulting to {MIN_BLOCK_LENGTH}."
            )
            return MIN_BLOCK_LENGTH

        if v > block_length_sampler.avg_block_length:
            # Adjust min_block_length to sampler's avg_block_length if it exceeds
            adjusted_min_block_length = block_length_sampler.avg_block_length
            warnings.warn(
                f"'min_block_length' ({v}) should be <= 'avg_block_length' "
                f"({block_length_sampler.avg_block_length}). "
                f"Setting to {adjusted_min_block_length}.",
                stacklevel=2,
            )
            logger.warning(
                f"'min_block_length' ({v}) exceeds 'avg_block_length' "
                f"({block_length_sampler.avg_block_length}). "
                f"Adjusted to {adjusted_min_block_length}."
            )
            return adjusted_min_block_length

        logger.debug(f"'min_block_length' validated: {v}")
        return v

    @model_validator(mode="after")
    def check_consistency(self) -> BlockGenerator:
        """
        Perform inter-field validation to ensure consistency among fields.

        This validator runs after all field validators have processed their respective fields,
        ensuring that interdependent fields maintain logical consistency.

        Returns
        -------
        BlockGenerator
            The validated BlockGenerator instance.

        Raises
        ------
        ValueError
            If any of the consistency checks fail.
        """
        input_length: int = self.input_length
        sampler: BlockLengthSampler = self.block_length_sampler
        overlap_length: int = self.overlap_length  # type: ignore
        min_block_length: int = self.min_block_length  # type: ignore

        if sampler.avg_block_length > input_length:
            error_msg = (
                f"'sampler.avg_block_length' ({sampler.avg_block_length}) must be <= "
                f"'input_length' ({input_length})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if overlap_length >= input_length:
            error_msg = (
                f"'overlap_length' ({overlap_length}) must be less than "
                f"'input_length' ({input_length})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if min_block_length > sampler.avg_block_length:
            error_msg = (
                f"'min_block_length' ({min_block_length}) cannot exceed "
                f"'avg_block_length' ({sampler.avg_block_length})."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug("All inter-field consistency checks passed.")
        return self

    def _create_block(self, start_index: int, block_length: int) -> np.ndarray:
        """
        Create a block of indices.

        Parameters
        ----------
        start_index : int
            Starting index of the block.
        block_length : int
            Length of the block.

        Returns
        -------
        np.ndarray
            An array representing the indices of a block in the time series.
        """
        end_index = start_index + block_length

        if end_index <= self.input_length:
            # Block does not exceed input_length; no wrapping needed
            logger.debug(
                f"Creating block from {start_index} to {end_index} without wrapping."
            )
            return np.arange(start_index, end_index)
        else:
            if self.wrap_around_flag:
                # Wrap around to the beginning
                end_index_wrapped = end_index % self.input_length
                logger.debug(
                    f"Creating block with wrap-around: {start_index} to {self.input_length} and 0 to {end_index_wrapped}."
                )
                return np.concatenate(
                    (
                        np.arange(start_index, self.input_length),
                        np.arange(0, end_index_wrapped),
                    )
                )
            else:
                # Adjust block_length to fit within input_length without wrapping
                adjusted_end_index = self.input_length
                logger.debug(
                    f"Creating block from {start_index} to {adjusted_end_index} without wrapping."
                )
                return np.arange(start_index, adjusted_end_index)

    def _calculate_start_index(self) -> int:
        """
        Calculate the starting index of a block.

        Returns
        -------
        int
            The starting index of the block.
        """
        if self.wrap_around_flag:
            start = self.rng.integers(self.input_length)  # type: ignore
            logger.debug(
                f"Wrap-around enabled. Randomly selected starting index: {start}."
            )
            return start
        else:
            logger.debug("Wrap-around disabled. Starting index set to 0.")
            return 0

    def _calculate_overlap_length(self, sampled_block_length: int) -> int:
        """
        Calculate the overlap length for a block.

        Parameters
        ----------
        sampled_block_length : int
            The length of the sampled block.

        Returns
        -------
        int
            The calculated overlap length.
        """
        if self.overlap_length is None:
            # Default to half of the sampled block length if overlap_length is not provided
            calculated_overlap = sampled_block_length // 2
            logger.debug(
                f"'overlap_length' not provided. Calculated default overlap: {calculated_overlap}."
            )
            return calculated_overlap
        else:
            # Ensure overlap_length is at least 1 and less than block_length
            calculated_overlap = min(
                # type: ignore
                max(self.overlap_length, 1),
                sampled_block_length - 1,
            )
            logger.debug(f"Calculated overlap length: {calculated_overlap}.")
            return calculated_overlap

    def _get_total_length_covered(
        self, block_length: int, overlap_length: int
    ) -> int:
        """
        Get the total length covered in the time series considering the current block length and overlap length.

        Parameters
        ----------
        block_length : int
            The length of the current block.
        overlap_length : int
            The overlap length between the current and next block.

        Returns
        -------
        int
            The total length covered so far.
        """
        total_length = block_length - overlap_length
        logger.debug(f"Total length covered increased by {total_length}.")
        return total_length

    def _get_next_block_length(
        self, sampled_block_length: int, total_length_covered: int
    ) -> int:
        """
        Get the next block length after considering wrap-around and total length covered.

        Parameters
        ----------
        sampled_block_length : int
            The sampled block length from the block length sampler.
        total_length_covered : int
            The total length covered so far.

        Returns
        -------
        int
            The adjusted block length.
        """
        if not self.wrap_around_flag:
            # Ensure block_length does not exceed remaining input length
            remaining_length = self.input_length - total_length_covered  # type: ignore
            adjusted_block_length = min(sampled_block_length, remaining_length)
            logger.debug(
                f"Wrap-around disabled. Adjusted block length from {sampled_block_length} to {adjusted_block_length} based on remaining length {remaining_length}."
            )
            return adjusted_block_length
        else:
            # Wrap-around is allowed; use sampled block length as is
            logger.debug(
                f"Wrap-around enabled. Using sampled block length: {sampled_block_length}."
            )
            return sampled_block_length

    def _calculate_next_start_index(
        self,
        start_index: int,
        block_length: int,
        overlap_length: int,
    ) -> int:
        """
        Calculate the next start index for generating the subsequent block.

        Parameters
        ----------
        start_index : int
            The start index of the current block.
        block_length : int
            The length of the current block.
        overlap_length : int
            The overlap length between the current and next block.

        Returns
        -------
        int
            The start index for the next block.
        """
        next_start_index = start_index + block_length - overlap_length
        next_start_index %= self.input_length
        logger.debug(f"Next start index calculated: {next_start_index}.")
        return next_start_index

    def generate_non_overlapping_blocks(self) -> list[np.ndarray]:
        """
        Generate non-overlapping block indices in the time series.

        Returns
        -------
        list[np.ndarray]
            list of numpy arrays containing the indices for each non-overlapping block.

        Raises
        ------
        ValueError
            If the block length sampler is not set.
        """
        logger.info("Generating non-overlapping blocks.")
        block_indices = []
        start_index = self._calculate_start_index()
        total_length = 0

        while total_length < self.input_length:  # type: ignore
            sampled_block_length = (
                self.block_length_sampler.sample_block_length()
            )
            logger.debug(f"Sampled block length: {sampled_block_length}.")

            block_length = self._get_next_block_length(
                sampled_block_length, total_length
            )
            logger.debug(f"Adjusted block length: {block_length}.")

            block = self._create_block(start_index, block_length)
            logger.debug(f"Generated block: {block}.")

            block_indices.append(block)
            total_length += block_length
            logger.debug(f"Total length covered so far: {total_length}.")

            start_index = self._calculate_next_start_index(
                start_index, block_length, overlap_length=0
            )

        validate_block_indices(block_indices, self.input_length)  # type: ignore
        logger.info("Non-overlapping block generation completed successfully.")
        return block_indices

    def generate_overlapping_blocks(self) -> list[np.ndarray]:
        """
        Generate overlapping block indices in the time series.

        Returns
        -------
        list[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.

        Notes
        -----
        The block indices are generated as follows:

        1. A starting index is determined based on the `wrap_around_flag`.
        2. A block length is sampled from the `block_length_sampler`.
        3. An overlap length is calculated from the block length.
        4. A block is created from the starting index and block length.
        5. The starting index is updated to the next starting index.
        6. Steps 2-5 are repeated until the total length covered equals the length of the time series.

        The block length sampler is used to sample the block length. The overlap length is calculated based on the sampled block length.
        The starting index is updated by adding the block length and subtracting the overlap length, then taking modulo `input_length` to ensure it wraps around if necessary.
        """
        logger.info("Generating overlapping blocks.")
        block_indices = []
        start_index = self._calculate_start_index()
        total_length_covered = 0
        start_indices = set()

        while total_length_covered < self.input_length:  # type: ignore
            if start_index in start_indices:
                # Detect repeating start index to prevent infinite loops
                warnings.warn(
                    "Detected repeating start index. Terminating block generation to prevent infinite loop.",
                    stacklevel=2,
                )
                logger.warning(
                    "Repeating start index detected. Terminating block generation."
                )
                break

            start_indices.add(start_index)
            sampled_block_length = (
                self.block_length_sampler.sample_block_length()
            )
            logger.debug(f"Sampled block length: {sampled_block_length}.")

            block_length = self._get_next_block_length(
                sampled_block_length, total_length_covered
            )
            logger.debug(f"Adjusted block length: {block_length}.")

            if block_length < self.min_block_length:  # type:ignore
                # Terminate if the block length is below the minimum required
                logger.debug(
                    f"Block length {block_length} is less than minimum {self.min_block_length}. Terminating."
                )
                break

            overlap_length = self._calculate_overlap_length(block_length)
            logger.debug(f"Calculated overlap length: {overlap_length}.")

            block = self._create_block(start_index, block_length)
            logger.debug(f"Generated block: {block}.")

            block_indices.append(block)
            total_length_covered += self._get_total_length_covered(
                block_length, overlap_length
            )
            logger.debug(
                f"Total length covered so far: {total_length_covered}."
            )

            start_index = self._calculate_next_start_index(
                start_index, block_length, overlap_length
            )

        # Prevent exceeding the maximum allowed iterations
        if total_length_covered >= self.input_length:
            logger.info("Overlapping block generation completed successfully.")
        else:
            warnings.warn(
                "Maximum iterations reached during block generation. Some input may remain uncovered.",
                stacklevel=2,
            )
            logger.warning(
                "Maximum iterations reached. Some input may remain uncovered."
            )

        validate_block_indices(block_indices, self.input_length)  # type: ignore
        return block_indices

    def generate_blocks(self, overlap_flag: bool = False) -> list[np.ndarray]:
        """
        Generate block indices based on the overlap flag.

        This method serves as a general entry point to generate either overlapping
        or non-overlapping blocks based on the provided flag.

        Parameters
        ----------
        overlap_flag : bool, optional
            If True, generate overlapping blocks. Otherwise, generate non-overlapping blocks. Default is False.

        Returns
        -------
        list[np.ndarray]
            list of numpy arrays representing the generated blocks.
        """
        if overlap_flag:
            return self.generate_overlapping_blocks()
        else:
            return self.generate_non_overlapping_blocks()
