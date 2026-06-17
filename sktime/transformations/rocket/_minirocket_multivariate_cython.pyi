"""Type stubs for the compiled MiniRocketMultivariate Cython kernels."""

import numpy as np
from numpy.typing import NDArray

def transform(
    X: NDArray[np.float32],
    num_channels_per_combination: NDArray[np.int32],
    channel_indices: NDArray[np.int32],
    dilations: NDArray[np.int32],
    num_features_per_dilation: NDArray[np.int32],
    biases: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def fit_biases(
    X: NDArray[np.float32],
    num_channels_per_combination: NDArray[np.int32],
    channel_indices: NDArray[np.int32],
    dilations: NDArray[np.int32],
    num_features_per_dilation: NDArray[np.int32],
    instance_indices: NDArray[np.int32],
) -> NDArray[np.float32]: ...
