"""Time Convolutional Neural Network (CNN) in PyTorch."""

__authors__ = ["sabasiddique1"]
__all__ = ["CNNNetworkTorch"]

import numpy as np

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class CNNNetworkTorch(NNModule):
    """Establish the network structure for a CNN in PyTorch.

    Zhao et al. 2017 uses sigmoid activation in the hidden layers.
    To obtain same behaviour as Zhao et al. 2017, set activation to "sigmoid".

    Adapted from the TensorFlow implementation in sktime.networks.cnn.

    Parameters
    ----------
    input_size : int or tuple
        Number of expected features in the input (n_dims), or tuple
        (n_instances, n_dims, series_length) for inferring n_dims.
    num_classes : int
        Number of classes to predict (classification) or 1 for regression.
    kernel_size : int, default = 7
        Length of the 1D convolution window.
    avg_pool_size : int, default = 3
        Size of the average pooling windows.
    n_conv_layers : int, default = 2
        Number of convolutional plus average pooling layers.
    filter_sizes : array-like of int, default = None
        Filter sizes per conv layer. If None, uses [6, 12].
        Extended to match n_conv_layers if shorter.
    activation : str or None, default = None
        Activation function used in the fully connected output layer.
        Supported: ['sigmoid', 'softmax', 'logsoftmax', 'logsigmoid', 'linear'].
        If None, no activation is applied.
    activation_hidden : str, default = "sigmoid"
        Activation function for hidden conv layers. Supported: ['relu', 'sigmoid'].
        Same default as TensorFlow CNN implementation.
    padding : str, default = "auto"
        Padding logic for conv layers.
        - "auto": "same" if series_length < 60, else "valid".
        - "valid", "same": passed to Conv1d.
    series_length : int or None, default = None
        Length of input series. Required when padding="auto" to compute fc input
        size. If None and padding="auto", defaults to 60 for flat size.
    bias : bool, default = True
        Whether conv and linear layers use a bias vector.
    random_state : int, default = 0
        Seed for reproducibility.

    References
    ----------
    .. [1] Zhao et al. Convolutional neural networks for time series classification,
    Journal of Systems Engineering and Electronics 28(1), 162--169, 2017
    """

    _tags = {
        "authors": ["sabasiddique1"],
        "maintainers": ["sabasiddique1"],
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size,
        num_classes,
        kernel_size=7,
        avg_pool_size=3,
        n_conv_layers=2,
        filter_sizes=None,
        activation=None,
        activation_hidden="sigmoid",
        padding="auto",
        series_length=None,
        bias=True,
        random_state=0,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.avg_pool_size = avg_pool_size
        self.n_conv_layers = n_conv_layers
        self.filter_sizes = [6, 12] if filter_sizes is None else list(filter_sizes)
        self.activation = activation
        self.activation_hidden = activation_hidden
        self.padding = padding
        self.series_length = series_length
        self.bias = bias
        self.random_state = random_state
        super().__init__()

        if isinstance(self.input_size, int):
            in_channels = self.input_size
        elif isinstance(self.input_size, tuple) and len(self.input_size) == 3:
            in_channels = self.input_size[1]
        else:
            raise ValueError(
                "input_size must be int (n_dims) or tuple (n_instances, n_dims, "
                f"series_length). Got {self.input_size}"
            )

        fs = self.filter_sizes
        nconv = self.n_conv_layers
        filter_sizes_list = fs[:nconv] + [fs[-1]] * max(0, nconv - len(fs))

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnAvgPool1d = _safe_import("torch.nn.AvgPool1d")
        nnFlatten = _safe_import("torch.nn.Flatten")
        nnLinear = _safe_import("torch.nn.Linear")
        nnModuleList = _safe_import("torch.nn.ModuleList")

        conv_layers = []
        pool_layers = []
        prev_channels = in_channels
        for i in range(n_conv_layers):
            conv = nnConv1d(
                in_channels=prev_channels,
                out_channels=filter_sizes_list[i],
                kernel_size=self.kernel_size,
                padding=0,
                bias=self.bias,
            )
            pool = nnAvgPool1d(kernel_size=self.avg_pool_size)
            conv_layers.append(conv)
            pool_layers.append(pool)
            prev_channels = filter_sizes_list[i]

        self._conv_modules = nnModuleList(conv_layers)
        self._pool_modules = nnModuleList(pool_layers)
        self.flatten = nnFlatten()

        self._activation_hidden = self._get_hidden_activation(activation_hidden)
        if self.activation:
            self._activation_out = self._instantiate_activation()
        else:
            self._activation_out = None
        self._padding_mode = padding
        self._filter_sizes_list = filter_sizes_list

        flat_size = self._compute_flat_size()
        self.fc = nnLinear(flat_size, num_classes, bias=self.bias)

    def _get_hidden_activation(self, name):
        if name == "relu":
            return _safe_import("torch.nn.ReLU")()
        if name == "sigmoid":
            return _safe_import("torch.nn.Sigmoid")()
        raise ValueError(
            f"activation_hidden must be 'relu' or 'sigmoid'. Got {name}"
        )

    def _instantiate_activation(self):
        if self.activation is None:
            return None
        if isinstance(self.activation, NNModule):
            return self.activation
        name = self.activation.lower() if isinstance(self.activation, str) else ""
        if name == "sigmoid":
            return _safe_import("torch.nn.Sigmoid")()
        if name == "softmax":
            return _safe_import("torch.nn.Softmax")(dim=1)
        if name == "logsoftmax":
            return _safe_import("torch.nn.LogSoftmax")(dim=1)
        if name == "logsigmoid":
            return _safe_import("torch.nn.LogSigmoid")()
        if name == "linear":
            return _safe_import("torch.nn.Identity")()
        raise ValueError(
            "activation must be one of 'sigmoid', 'softmax', 'logsoftmax', "
            f"'logsigmoid', 'linear' or None. Got {self.activation}"
        )

    def _compute_flat_size(self):
        """Compute flattened size after conv+pool stack (same padding logic as forward)."""
        length = (
            60
            if self.series_length is None
            else self.series_length
        )
        padding_val = self._get_padding_value(length)
        for _ in range(self.n_conv_layers):
            if padding_val > 0:
                length = length + 2 * padding_val
            length = length - self.kernel_size + 1
            length = (length - self.avg_pool_size) // self.avg_pool_size + 1
        return length * self._filter_sizes_list[-1]

    def _get_padding_value(self, series_length):
        if self._padding_mode == "same":
            return (self.kernel_size - 1) // 2
        if self._padding_mode == "valid":
            return 0
        if self._padding_mode == "auto":
            return (self.kernel_size - 1) // 2 if series_length < 60 else 0
        return 0

    def forward(self, X):
        """Forward pass.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, seq_length, n_dims)
            Input time series.

        Returns
        -------
        out : torch.Tensor
            Output logits (batch_size, num_classes) or (batch_size, 1).
        """
        if isinstance(X, np.ndarray):
            X = _safe_import("torch.from_numpy")(X).float()
        x = X.transpose(1, 2)
        batch_size, _, series_length = x.shape
        padding_val = self._get_padding_value(series_length)
        for i in range(self.n_conv_layers):
            if padding_val > 0:
                x = _safe_import("torch.nn.functional.pad")(
                    x, (padding_val, padding_val), mode="constant", value=0
                )
            x = self._conv_modules[i](x)
            x = self._activation_hidden(x)
            x = self._pool_modules[i](x)
        x = self.flatten(x)
        out = self.fc(x)
        if self._activation_out is not None:
            out = self._activation_out(out)
        return out
