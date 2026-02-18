"""Fully Connected Neural Network (FCN) deep learning network structure in Torch.


"""

__authors__ = ["RecreationalMath"]
__all__ = ["FCNNetworkTorch"]


from collections.abc import Callable

import numpy as np

from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
NNModule = _safe_import("torch.nn.Module")


class FCNNetworkTorch(NNModule):
    """Establish the network structure for a FCN in PyTorch.

    Adapted from the implementation of Fawaz et. al
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py

    Implements network in [1]_.

    Parameters
    ----------
    input_shape : tuple
        shape of the input data fed into the network
    num_classes : int
        Number of classes to predict
    n_layers : int, default = 3
        Number of convolution layers. The original FCN architecture uses 3 layers.
    n_filters : int or list of int, default = [128, 256, 128]
        Number of filters used in convolution layers. If int, the same number of
        filters is used for all layers. If list, must have length equal to n_layers.
    kernel_sizes : int or list of int, default = [8, 5, 3]
        Kernel size used in convolution layers. If int, the same kernel size
        is used for all layers. If list, must have length equal to n_layers.
    activation : str or None or an instance of activation functions defined in
        torch.nn, default = "relu"
        Activation function used in the fully connected output layer. List of supported
        activation functions: ['sigmoid', 'softmax', 'logsoftmax', 'logsigmoid'].
        If None, then no activation function is applied.
    activation_hidden : str or None or an instance of activation functions defined in
        torch.nn, default = "relu"
        The activation function applied inside the hidden layers of the FCN.
        Can be any of "relu", "leakyrelu", "elu", "prelu", "gelu", "selu",
        "rrelu", "celu", "tanh", "hardtanh".
    random_state   : int, default = 0
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
        "authors": ["Wvidit"],
        "maintainers": ["Wvidit"],
        "python_dependencies": ["torch"],
        "capability:random_state": True,
        "property:randomness": "stochastic",
    }

    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        n_layers: int = 3,
        n_filters: int | list[int] = (128, 256, 128),
        kernel_sizes: int | list[int] = (8, 5, 3),
        activation: str | None | Callable = None,
        activation_hidden: str | None | Callable = "relu",
        random_state: int = 0,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.random_state = random_state
        super().__init__()

        # checking n_filters list length
        if isinstance(self.n_filters, list) or isinstance(self.n_filters, tuple):
            if len(self.n_filters) != self.n_layers:
                raise ValueError(
                    "If `n_filters` is a list/tuple, it must have length equal to "
                    f"`n_layers`. Found length of `n_filters` to be {len(self.n_filters)}"
                    f" and `n_layers` to be {self.n_layers}."
                )
        else:
             # Broadcast int to list
            self.n_filters = [self.n_filters] * self.n_layers

        # checking kernel_sizes list length
        if isinstance(self.kernel_sizes, list) or isinstance(self.kernel_sizes, tuple):
            if len(self.kernel_sizes) != self.n_layers:
                raise ValueError(
                    "If `kernel_sizes` is a list/tuple, it must have length equal to "
                    f"`n_layers`. Found length of `kernel_sizes` to be {len(self.kernel_sizes)}"
                    f" and `n_layers` to be {self.n_layers}."
                )
        else:
             # Broadcast int to list
            self.kernel_sizes = [self.kernel_sizes] * self.n_layers

        # Checking input dimensions
        if isinstance(self.input_shape, tuple):
            if len(self.input_shape) == 3:
                # Shape is (n_instances, n_channels, series_length)
                self.n_channels = self.input_shape[1]
            else:
                raise ValueError(
                    "If `input_shape` is a tuple, it must be of length 3 and in "
                    "format (n_instances, n_channels, series_length). "
                    f"Found length of {len(self.input_shape)}"
                )
        else:
            raise TypeError(
                "`input_shape` should be of type tuple. "
                f"But found the type to be: {type(self.input_shape)}"
            )

        # defining the model architecture
        layers = []
        
        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")

        in_channels = self.n_channels

        for i in range(self.n_layers):
            layers.append(
                nnConv1d(
                    in_channels=in_channels,
                    out_channels=self.n_filters[i],
                    kernel_size=self.kernel_sizes[i],
                    padding="same"
                )
            )
            layers.append(nnBatchNorm1d(num_features=self.n_filters[i]))
            
            if self.activation_hidden:
                layers.append(self._instantiate_activation(layer="hidden"))
            
            in_channels = self.n_filters[i]

        nnSequential = _safe_import("torch.nn.Sequential")
        self.backbone = nnSequential(*layers)

        # defining the Global Average Pooling and Output layer
        # Global Average Pooling is performed in forward() typically in Torch 
        # via mean(), but we establish the FC layer here.
        nnLinear = _safe_import("torch.nn.Linear")
        
        self.fc = nnLinear(
            in_features=self.n_filters[-1], # Matches output of last conv
            out_features=self.num_classes,
        )

        if self.activation:
            self._activation = self._instantiate_activation(layer="output")

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor of shape (n_instances, n_channels, series_length)
            Input tensor containing the time series data.

        Returns
        ---------
        out : torch.Tensor of shape (n_instances, num_classes)
            Output probabilities or logits.
        """
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()

        out = self.backbone(X)
        out = out.mean(dim=-1)

        out = self.fc(out)
        if self.activation:
            out = self._activation(out)

        return out

    def _instantiate_activation(self, layer):
        """Instantiate the activation function to be applied within the FCN.

        Parameters
        ----------
        layer : str
            The name of the layer for which to instantiate the activation function.
            Can be either 'output' or 'hidden'.

        Returns
        -------
        activation_function : torch.nn.Module
            The activation function to be applied on the output layer.
        """
        if layer == "hidden":
            if isinstance(self.activation_hidden, NNModule):
                return self.activation_hidden
            elif isinstance(self.activation_hidden, str):
                if self.activation_hidden.lower() == "relu":
                    return _safe_import("torch.nn.ReLU")()
                elif self.activation_hidden.lower() == "leakyrelu":
                    return _safe_import("torch.nn.LeakyReLU")()
                elif self.activation_hidden.lower() == "elu":
                    return _safe_import("torch.nn.ELU")()
                elif self.activation_hidden.lower() == "prelu":
                    return _safe_import("torch.nn.PReLU")()
                elif self.activation_hidden.lower() == "gelu":
                    return _safe_import("torch.nn.GELU")()
                elif self.activation_hidden.lower() == "selu":
                    return _safe_import("torch.nn.SELU")()
                elif self.activation_hidden.lower() == "rrelu":
                    return _safe_import("torch.nn.RReLU")()
                elif self.activation_hidden.lower() == "celu":
                    return _safe_import("torch.nn.CELU")()
                elif self.activation_hidden.lower() == "tanh":
                    return _safe_import("torch.nn.Tanh")()
                elif self.activation_hidden.lower() == "hardtanh":
                    return _safe_import("torch.nn.Hardtanh")()
                else:
                    raise ValueError(
                        "If `activation_hidden` is not None, it must be one of "
                        "'relu', 'leakyrelu', 'elu', 'prelu', 'gelu', 'selu', "
                        "'rrelu', 'celu', 'tanh', 'hardtanh'. "
                        f"But found {self.activation_hidden}"
                    )
            else:
                raise TypeError(
                    "`activation_hidden` should either be of type torch.nn.Module or"
                    f" str. But found the type to be: {type(self.activation_hidden)}"
                )
        elif layer == "output":
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
