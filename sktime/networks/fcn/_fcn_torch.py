"""Fully Convolutional Network (FCN) for time series classification in PyTorch."""

__authors__ = ["Ali-John"]

import numpy as np

from sktime.utils.dependencies import _safe_import
# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")

class FCNNetworkTorch(NNModule):
    """Establish the network structure for a FCN in PyTorch.

    Implements network in [1].

    Parameters
    ----------
    input_size : int
        Number of expected features in the input (number of dimensions/channels)
    num_classes : int
        Number of classes to predict
    activation : str or None or an instance of activation functions defined in torch.nn, default = None
        Activation function used in the fully connected output layer.
        List of supported activation functions: 'sigmoid', 'softmax',
        'logsoftmax', 'logsigmoid'. If None, no activation is applied.
    random_state : int, default = 0
        Seed to ensure reproducibility.

    References
    ----------
    .. [1] Network originally defined in:
    @inproceedings{wang2017time,
    title={Time series classification from scratch with deep neural networks:
    A strong baseline},
    author={Wang, Zhiguang and Yan, Weizhong and Oates, Tim},
    booktitle={2017 International joint conference on neural networks
    (IJCNN)},
    pages={1578--1585},
    year={2017},
    organization={IEEE}
    }
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["Ali-John"],
        "maintainers": ["Ali-John"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        activation: str | None = None,
        random_state: int = 0,
    ):
        self.num_classes = num_classes
        self.activation = activation
        self.random_state = random_state

        super().__init__()
        
        if isinstance(input_size, int):
            input_features = input_size
        elif isinstance(input_size, tuple):
            if len(input_size) == 3:
                input_features = input_size[1]
            else:
                raise ValueError(
                    "If `input_size` is a tuple, it must either be of length 3 and in "
                    "format (n_instances, n_dims, series_length). "
                    f"Found length of {len(input_size)}"
                )
        else:
            raise TypeError(
                "`input_size` should either be of type int or tuple. "
                f"But found the type to be: {type(input_size)}"
            )
        
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnReLU = _safe_import("torch.nn.ReLU")
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")
        nnLinear = _safe_import("torch.nn.Linear")

        # activation
        self.relu = nnReLU()
        # global average pooling layer
        self.gap = nnAdaptiveAvgPool1d(1)

        # convolutional block 1: 128 filters, kernel size 8
        self.conv1 = nnConv1d(in_channels=input_features, out_channels=128, kernel_size=8, padding='same')
        self.bn1 = nnBatchNorm1d(num_features=128)

        # convolutional block 2: 256 filters, kernel size 5
        self.conv2 = nnConv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.bn2 = nnBatchNorm1d(num_features=256)

        # convolutional block 3: 128 filters, kernel size 3
        self.conv3 = nnConv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.bn3 = nnBatchNorm1d(num_features=128)

        # fully connected layer
        self.fc = nnLinear(in_features=128, out_features=self.num_classes)

        # activation for output layer if provided
        if self.activation:
            self._activation = self._instantiate_activation()
    
    def forward(self, X):
        """Define the forward pass of the network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_length, input_size)
            Input data to the network

        Returns
        -------
        torch.Tensor of shape (batch_size, num_classes)
            Output of the network before the final classification layer
        """
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()

        # PyTorch Conv1d expects input of shape (batch, channels, length)
        # DataLoader provides (batch, length, channels), so we transpose
        X = X.transpose(1, 2)

        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.gap(out)
        out = out.squeeze(-1)  # remove last dimension

        # fully connected layer
        out = self.fc(out)

        # apply output activation if specified 
        if self.activation:
            out = self._activation(out)

        return out

    def _instantiate_activation(self):
        """Instantiate the activation function for the output layer.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the output layer.
        """
        if isinstance(self.activation, NNModule):
            return self.activation
        elif isinstance(self.activation, str):
            if self.activation.lower() == "sigmoid":
                return _safe_import("torch.nn.Sigmoid")()
            elif self.activation.lower() == "softmax":
                return _safe_import("torch.nn.Softmax")(dim=1)
            elif self.activation.lower() == "logsoftmax":
                return _safe_import("torch.nn.LogSoftmax")(dim=1)
            elif self.activation.lower() == "logsigmoid":
                return _safe_import("torch.nn.LogSigmoid")()
            else:
                raise ValueError(
                    "If `activation` is not None, it must be one of "
                    "'sigmoid', 'logsigmoid', 'softmax' or 'logsoftmax'. "
                    f"Found {self.activation}"
                )
        else:
            raise TypeError(
                "`activation` should either be of type str or torch.nn.Module. "
                f"But found the type to be: {type(self.activation)}"
            )

