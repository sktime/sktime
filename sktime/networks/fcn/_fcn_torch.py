"""Fully Connected Neural Network (FCN) for classification and regression in PyTorch."""

__authors__ = ["kajal-jotwani"]
__all__ = ["FCNNetworkTorch"]

import numpy as np

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")

class FCNNetworkTorch(NNModule):
    """Fully Convolutional Network (FCN) for classification and regression in PyTorch.

    Parameters
    ----------
    input_size : int or tuple
        Number of expected input features. If tuple, must be of shape
        (n_instances, n_dims, series_length).

    num_classes : int
        Number of output classes. For regression, this is typically 1.

    n_filters : tuple of int, default = (128, 256, 128)
        Number of convolutional filters for each convolutional layer.

    kernel_sizes : tuple of int, default = (8, 5, 3)
        Kernel sizes for each convolutional layer.

    use_batch_norm : bool, default = True
        Whether to apply batch normalization after each convolution.

    random_state : int, default = 0
        Seed to ensure reproducibility.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["kajal-jotwani"],
        "maintainers": ["kajal-jotwani"],
        "python_version": ">=3.10",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_filters=(128, 256, 128),
        kernel_sizes=(8, 5, 3),
        use_batch_norm: bool = True,
        random_state: int = 0,
    ):
        self.random_state = random_state
        self.num_classes = num_classes
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.use_batch_norm = use_batch_norm

        super().__init__()

        # Handle input size
        if isinstance(input_size, int):
            in_channels = input_size
        elif isinstance(input_size, tuple):
            if len(input_size) == 3:
                in_channels = input_size[1]
            else:
                raise ValueError(
                    "If `input_size` is a tuple, it must be of length 3 and in "
                    "format (n_instances, n_dims, series_length)."
                )
        else:
            raise TypeError(
                "`input_size` should be of type int or tuple. "
                f"Found type {type(input_size)}"
            )

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnReLU = _safe_import("torch.nn.ReLU")
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        nnLinear = _safe_import("torch.nn.Linear")

        # Conv block 1
        self.conv1 = nnConv1d(
            in_channels=in_channels,
            out_channels=n_filters[0],
            kernel_size=kernel_sizes[0],
            padding="same",
        )
        self.bn1 = nnBatchNorm1d(n_filters[0]) if use_batch_norm else None

        # Conv block 2
        self.conv2 = nnConv1d(
            in_channels=n_filters[0],
            out_channels=n_filters[1],
            kernel_size=kernel_sizes[1],
            padding="same",
        )
        self.bn2 = nnBatchNorm1d(n_filters[1]) if use_batch_norm else None

        # Conv block 3
        self.conv3 = nnConv1d(
            in_channels=n_filters[1],
            out_channels=n_filters[2],
            kernel_size=kernel_sizes[2],
            padding="same",
        )
        self.bn3 = nnBatchNorm1d(n_filters[2]) if use_batch_norm else None

        self.relu = nnReLU()

        # Global Average Pooling
        self.global_avg_pool = nnAdaptiveAvgPool1d(1)

        # Output layer
        self.fc = nnLinear(n_filters[2], num_classes)

    def forward(self, X):
        """Forward pass through the FCN network."""
        if isinstance(X, np.ndarray):
            torch_from_numpy = _safe_import("torch.from_numpy")
            X = torch_from_numpy(X).float()

        # Expect shape (batch, channels, series_length)
        x = self.conv1(X)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        if self.bn3 is not None:
            x = self.bn3(x)
        x = self.relu(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        x = self.fc(x)

        return x