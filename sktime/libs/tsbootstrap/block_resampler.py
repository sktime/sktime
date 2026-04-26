import warnings
from collections.abc import Callable
from numbers import Integral

import numpy as np
from numpy.random import Generator

from tsbootstrap.utils.types import RngTypes
from tsbootstrap.utils.validate import (
    validate_block_indices,
    validate_rng,
    validate_weights,
)


class BlockResampler:
    """
    A class to perform block resampling.

    Methods
    -------
    resample_blocks()
        Resamples blocks and their corresponding tapered_weights with replacement to create a new list of blocks and tapered_weights with total length equal to n.
    resample_block_indices_and_data()
        Generate block indices and corresponding data for the input data array X.
    """

    def __init__(
        self,
        blocks,
        X: np.ndarray,
        block_weights=None,
        tapered_weights: Callable = None,
        rng: RngTypes = None,
    ):
        """
        Initialize the BlockResampler with the selected distribution and average block length.

        Parameters
        ----------
        blocks : List[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.
        X : np.ndarray
            The input data array.
        block_weights : Union[np.ndarray, Callable], optional
            An array of weights or a callable function to generate weights. If None, then the default uniform weights are used.
        tapered_weights : Union[np.ndarray, Callable], optional
            An array of weights to apply to the data within the blocks. If None, then the default uniform weights are used.
        rng : np.random.Generator, optional
            Generator for reproducibility. If None, the global random state is used.
        """
        self.X = X
        self.blocks = blocks
        self.rng = rng
        self.block_weights = block_weights
        self.tapered_weights = tapered_weights

    @property
    def X(self) -> np.ndarray:
        """The input data array."""
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        """
        Set the input data array.

        Parameters
        ----------
        value : np.ndarray
            The input data array.


        Raises
        ------
        TypeError
            If the input data array is not a numpy array.
        ValueError
            If the input data array has less than two elements or if it is not a 1D or 2D array.


        Notes
        -----
        If the input data array is a 1D array, then it is reshaped to a 2D array.

        Examples
        --------
        >>> import numpy as np
        >>> from block_resampler import BlockResampler
        >>> X = np.array([1, 2, 3, 4, 5])
        >>> block_resampler = BlockResampler(blocks=[[0, 1, 2], [3, 4]], X=X)
        >>> block_resampler.X
        array([[1],
                [2],
                [3],
                [4],
                [5]])
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("'X' must be a numpy array.")
        else:
            if value.size < 2:
                raise ValueError("'X' must have at least two elements.")
            elif value.ndim == 1:
                warnings.warn(
                    "Input 'X' is a 1D array. It will be reshaped to a 2D array.",
                    stacklevel=2,
                )
                value = value.reshape(-1, 1)
            elif value.ndim > 2:
                raise ValueError("'X' must be a 1D or 2D numpy array.")
        self._X = value

    @property
    def blocks(self):
        """A list of numpy arrays where each array represents the indices of a block in the time series."""
        return self._blocks

    @blocks.setter
    def blocks(self, value) -> None:
        """
        Set the list of blocks.

        Parameters
        ----------
        value : List[np.ndarray]
            A list of numpy arrays where each array represents the indices of a block in the time series.


        Raises
        ------
        TypeError
            If the list of blocks is not a list.
        ValueError
            If the list of blocks is empty or if it contains non-integer arrays.


        Notes
        -----
        The list of blocks is sorted in ascending order.
        """
        validate_block_indices(value, self.X.shape[0])  # type: ignore
        self._blocks = value

    @property
    def rng(self) -> Generator:
        """Generator for reproducibility."""
        return self._rng

    @rng.setter
    def rng(self, value: RngTypes) -> None:
        """
        Set the random number generator.

        Parameters
        ----------
        value : RngTypes
            Generator for reproducibility.


        Raises
        ------
        TypeError
            If the random number generator is not a numpy random Generator or an integer.
        ValueError
            If the random number generator is an integer but it is not a non-negative integer.
        """
        self._rng = validate_rng(value, allow_seed=True)

    @property
    def block_weights(self) -> np.ndarray:
        """An array of normalized block_weights."""
        return self._block_weights

    @block_weights.setter
    def block_weights(self, value) -> None:
        """
        Set the block_weights array.

        Parameters
        ----------
        value : Union[np.ndarray, Callable]
            An array of weights or a callable function to generate weights.
            If None, then the default uniform weights are used.


        Raises
        ------
        TypeError
            If the block_weights array is not a numpy array or a callable function.
        ValueError
            If the block_weights array is a numpy array but it is empty or if it contains non-integer arrays.
            If the block_weights array is a callable function but the output is not a 1D array of length 'size'.
        """
        self._block_weights = self._prepare_block_weights(value)

    @property
    def tapered_weights(self):
        """A list of normalized weights."""
        return self._tapered_weights

    @tapered_weights.setter
    def tapered_weights(self, value: Callable) -> None:
        """
        Set the tapered_weights array.

        Parameters
        ----------
        value : Optional[Callable]
            A callable function to generate weights.
            If None, then the default uniform weights are used.

        Raises
        ------
        TypeError
            If the tapered_weights array is not a callable function.
        ValueError
            If the tapered_weights array is a callable function but the output is not a 1D array of length 'size'.
        """
        self._tapered_weights = self._prepare_tapered_weights(value)

    @staticmethod
    def _normalize_array(array: np.ndarray) -> np.ndarray:
        """
        Normalize the weights array.

        Parameters
        ----------
        array : np.ndarray
            n-dimensional array.

        Returns
        -------
        np.ndarray
            An array of normalized values, with the same shape as the input array.
        """
        sum_array = np.sum(array, axis=0, keepdims=True)
        zero_mask = sum_array != 0
        normalized_array = np.where(
            zero_mask, array / sum_array, 1.0 / array.shape[0]
        )
        return normalized_array

    def _prepare_tapered_weights(self, tapered_weights: Callable = None):
        """
        Prepare the tapered weights array by normalizing it or generating it.

        Parameters
        ----------
        tapered_weights : Union[np.ndarray, Callable]
            An array of weights or a callable function to generate weights.
        size : int, optional
            The size of the weights array (required for "tapered_weights").
            If None, then the size is the same as the block length.

        Returns
        -------
        np.ndarray or List[np.ndarray]
            An array or list of normalized weights.
        """
        block_lengths = np.array([len(block) for block in self.blocks])
        size = block_lengths

        if callable(tapered_weights):
            tapered_weights_arr = self._handle_callable_weights(
                tapered_weights, size
            )
            # Ensure that the edges are not exactly 0, while ensure that the max weight stays the same.
            tapered_weights_arr = [
                np.maximum(weights, 0.1) for weights in tapered_weights_arr
            ]
            # Ensure that the maximum weight is 1.
            tapered_weights_arr = [
                weights / np.max(weights) for weights in tapered_weights_arr
            ]
        elif tapered_weights is None:
            tapered_weights_arr = [np.full(size_iter, 1) for size_iter in size]
        else:
            raise TypeError(
                f"{tapered_weights} must be a callable function or None."
            )

        for weights in tapered_weights_arr:
            validate_weights(weights)

        return tapered_weights_arr

    def _prepare_block_weights(self, block_weights=None) -> np.ndarray:
        """
        Prepare the block_weights array by normalizing it or generating it based on the callable function provided.

        Parameters
        ----------
        block_weights : Union[np.ndarray, Callable], optional
            An array of weights or a callable function to generate weights. Defaults to None.

        Returns
        -------
        np.ndarray
            An array of normalized block_weights.
        """
        size = self.X.shape[0]

        if callable(block_weights):
            block_weights_arr = self._handle_callable_weights(
                block_weights, size
            )
        elif isinstance(block_weights, np.ndarray):
            block_weights_arr = self._handle_array_block_weights(
                block_weights, size
            )
        elif block_weights is None:
            block_weights_arr = np.full(size, 1 / size)
        else:
            raise TypeError(
                "'block_weights' must be a numpy array or a callable function or None."
            )

        # Validate the block_weights array
        validate_weights(block_weights_arr)
        # Normalize the block_weights array
        block_weights_arr = self._normalize_array(block_weights_arr)

        return block_weights_arr

    def _handle_callable_weights(
        self,
        weights_func: Callable,
        size,
    ) -> np.ndarray:
        """
        Handle callable block_weights by executing the function and validating the output.

        Parameters
        ----------
        block_weights : Callable
            A callable function to generate block weights.
        size : int
            The size of the block_weights array.

        Returns
        -------
        np.ndarray
            An array of block_weights.
        """
        weights_arr = self._generate_weights_from_callable(weights_func, size)

        self._validate_callable_generated_weights(
            weights_arr, size, weights_func.__name__
        )

        return weights_arr

    def _generate_weights_from_callable(
        self,
        weights_func: Callable,
        size,
    ):
        """
        Generate weights from a callable function.

        Parameters
        ----------
        weights_func : Callable
            A callable function to generate weights.
        size : Union[Integral, List[Integral], np.ndarray]
            The size of the weights array.

        Returns
        -------
        np.ndarray
            An array of weights.
        """
        if isinstance(size, Integral):
            return weights_func(size)
        elif isinstance(size, (np.ndarray, list)):  # noqa: UP038
            return [weights_func(size_iter) for size_iter in size]
        else:
            raise TypeError(
                "size must be an integer or a list/array of integers"
            )

    def _validate_callable_generated_weights(
        self,
        weights_arr,
        size,
        callable_name: str,
    ):
        """
        Validate the output of a callable function that generates either block_weights or tapered_weights.

        Parameters
        ----------
        weights_arr : Union[np.ndarray, List[np.ndarray]]
            An array or list of arrays of weights.
        size : Union[Integral, List[Integral]]
            The size of the weights array.
        callable_name : str
            The name of the callable function.

        Raises
        ------
        TypeError
            If the output of the callable function is not a numpy array.
        ValueError
            If the output of the callable function is not a 1d array of length 'size'.

        Returns
        -------
        None
        """
        if isinstance(weights_arr, list):
            print("dealing with tapered_weights")
            weights_arr = weights_arr[0]
            size = size[0]
        if not isinstance(weights_arr, np.ndarray):
            raise TypeError(
                f"Output of '{callable_name}(size)' must be a numpy array."
            )
        if len(weights_arr) != size or weights_arr.ndim != 1:
            raise ValueError(
                f"Output of '{callable_name}(size)' must be a 1d array of length 'size'."
            )

    def _handle_array_block_weights(
        self, block_weights: np.ndarray, size: int
    ) -> np.ndarray:
        """
        Handle array block_weights by validating the array and returning it.

        Parameters
        ----------
        block_weights : np.ndarray
            An array of block_weights.
        size : int
            The size of the block_weights array.

        Returns
        -------
        np.ndarray
            An array of block_weights.
        """
        if block_weights.shape[0] == 0:
            return np.full(size, 1 / size)
        elif block_weights.shape[0] != size:
            raise ValueError(
                "block_weights array must have the same size as X"
            )
        return block_weights

    def resample_blocks(self):
        """
        Resample blocks and corresponding tapered weights with replacement to create a new list of blocks and tapered weights with total length equal to n.

        Returns
        -------
        Tuple[list of ndarray, list of ndarray]
            The newly generated list of blocks and their corresponding tapered_weights
            with total length equal to n.

        Example
        -------
        >>> block_resampler = BlockResampler(blocks=blocks, X=data)
        >>> new_blocks, new_tapered_weights = block_resampler.resample_blocks()
        >>> len(new_blocks) == len(data)
        True
        """
        n = self.X.shape[0]
        block_dict = {block[0]: block for block in self.blocks}
        tapered_weights_dict = {
            block[0]: weight
            for block, weight in zip(self.blocks, self.tapered_weights)
        }
        first_indices = np.array(list(block_dict.keys()))
        block_lengths = np.array([len(block) for block in self.blocks])
        block_weights = np.array(
            [self.block_weights[idx] for idx in first_indices]
        )

        new_blocks, new_tapered_weights, total_samples = [], [], 0
        while total_samples < n:
            eligible_mask = (block_lengths <= n - total_samples) & (
                block_weights > 0  # type: ignore
            )
            if not np.any(eligible_mask):
                incomplete_eligible_mask = (block_lengths > 0) & (
                    block_weights > 0  # type: ignore
                )
                incomplete_eligible_weights = block_weights[
                    incomplete_eligible_mask
                ]

                index = self.rng.choice(
                    first_indices[incomplete_eligible_mask],
                    p=incomplete_eligible_weights
                    / incomplete_eligible_weights.sum(),
                )
                selected_block = block_dict[index]
                selected_tapered_weights = tapered_weights_dict[index]
                new_blocks.append(selected_block[: n - total_samples])
                new_tapered_weights.append(
                    selected_tapered_weights[: n - total_samples]
                )
                break

            eligible_weights = block_weights[eligible_mask]
            index = self.rng.choice(
                first_indices[eligible_mask],
                p=eligible_weights / eligible_weights.sum(),
            )
            selected_block = block_dict[index]
            selected_tapered_weights = tapered_weights_dict[index]
            new_blocks.append(selected_block)
            new_tapered_weights.append(selected_tapered_weights)
            total_samples += len(selected_block)

        return new_blocks, new_tapered_weights

    def resample_block_indices_and_data(
        self,
    ):
        """
        Generate block indices and corresponding data for the input data array X.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing a list of block indices and a list of corresponding modified data blocks.

        Example
        -------
        >>> block_resampler = BlockResampler(blocks=blocks, X=data)
        >>> block_indices, block_data = block_resampler.resample_block_indices_and_data()
        >>> len(block_indices) == len(data)
        True

        Notes
        -----
        The block indices are generated using the following steps:
        1. Generate block weights using the block_weights argument.
        2. Resample blocks with replacement to create a new list of blocks with total length equal to n.
        3. Apply tapered_weights to the data within the blocks if provided.
        """
        (
            resampled_block_indices,
            resampled_tapered_weights,
        ) = self.resample_blocks()
        block_data = []

        for i, block in enumerate(resampled_block_indices):
            taper = resampled_tapered_weights[i]
            data_block = self.X[block]
            block_data.append(data_block * taper.reshape(-1, 1))

        return resampled_block_indices, block_data

    def __repr__(self) -> str:
        return f"BlockResampler(blocks={self.blocks}, X={self.X}, block_weights={self.block_weights}, tapered_weights={self.tapered_weights}, rng={self.rng})"

    def __str__(self) -> str:
        return f"BlockResampler with blocks of length {len(self.blocks)}, input data of shape {self.X.shape}, block weights {self.block_weights}, tapered weights {self.tapered_weights}, and random number generator {self.rng}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BlockResampler):
            return (
                self.blocks == other.blocks
                and np.array_equal(self.X, other.X)
                and self.block_weights == other.block_weights
                and self.tapered_weights == other.tapered_weights
                and self.rng == other.rng
            )
        return False
