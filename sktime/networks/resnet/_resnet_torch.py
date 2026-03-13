"""Residual Network (ResNet) in PyTorch (minus the final output layer)."""

__authors__ = ["dakshhhhh16"]
__all__ = ["ResNetNetworkTorch"]

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class ResNetNetworkTorch(NNModule):
    """Establish the network structure for a ResNet in PyTorch.

    Adapted from the TensorFlow implementation in
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    The network consists of 3 residual blocks, each with 3 Conv1D layers
    followed by BatchNorm and an activation function. Each block has a
    shortcut connection. The output is a Global Average Pooled representation
    followed by a dense output layer.

    Parameters
    ----------
    n_channels : int
        Number of input channels (n_dimensions of the time series).
    series_length : int
        Length of the input time series (number of timesteps).
    n_feature_maps : int, optional (default=64)
        Number of feature maps (filters) in the first residual block.
        Second and third blocks use ``n_feature_maps * 2``.
    activation_hidden : str, optional (default="relu")
        Activation function used in the hidden layers.
        Supported: ``"relu"``, ``"sigmoid"``, ``"logsigmoid"``.
    activation_output : str or None, optional (default=None)
        Activation function used in the output layer.
        If None, no activation is applied.
        Supported: ``"sigmoid"``, ``"softmax"``, ``"logsoftmax"``, ``"logsigmoid"``.
    use_bias : bool, optional (default=True)
        Whether bias should be included in the output Dense layer.
    num_classes : int, optional (default=1)
        Number of output classes. Use 1 for regression.
    random_state : int or None, optional (default=None)
        Seed for random number generation.

    References
    ----------
    .. [1] Wang et al, Time series classification from
    scratch with deep neural networks: A strong baseline,
    International joint conference on neural networks (IJCNN), 2017.
    """

    _tags = {
        "authors": ["hfawaz", "James-Large", "Withington", "nilesh05apr", "noxthot"],
        "maintainers": ["dakshhhhh16"],
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        n_channels,
        series_length,
        n_feature_maps=64,
        activation_hidden="relu",
        activation_output=None,
        use_bias=True,
        num_classes=1,
        random_state=None,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.series_length = series_length
        self.n_feature_maps = n_feature_maps
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.use_bias = use_bias
        self.num_classes = num_classes
        self.random_state = random_state

        torch = _safe_import("torch")
        if random_state is not None:
            torch.manual_seed(random_state)

        nnConv1d = _safe_import("torch.nn.Conv1d")
        nnBatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        nnLinear = _safe_import("torch.nn.Linear")
        nnSequential = _safe_import("torch.nn.Sequential")
        nnAdaptiveAvgPool1d = _safe_import("torch.nn.AdaptiveAvgPool1d")

        # ---- Residual Block 1 ----
        # input shape: (batch, n_channels, series_length)
        self.block1_conv1 = nnConv1d(
            n_channels, n_feature_maps, kernel_size=8, padding="same"
        )
        self.block1_bn1 = nnBatchNorm1d(n_feature_maps)
        self.block1_conv2 = nnConv1d(
            n_feature_maps, n_feature_maps, kernel_size=5, padding="same"
        )
        self.block1_bn2 = nnBatchNorm1d(n_feature_maps)
        self.block1_conv3 = nnConv1d(
            n_feature_maps, n_feature_maps, kernel_size=3, padding="same"
        )
        self.block1_bn3 = nnBatchNorm1d(n_feature_maps)

        # Shortcut for block 1: project input channels -> n_feature_maps
        self.block1_shortcut = nnSequential(
            nnConv1d(n_channels, n_feature_maps, kernel_size=1, padding="same"),
            nnBatchNorm1d(n_feature_maps),
        )

        # ---- Residual Block 2 ----
        nfm2 = n_feature_maps * 2
        self.block2_conv1 = nnConv1d(
            n_feature_maps, nfm2, kernel_size=8, padding="same"
        )
        self.block2_bn1 = nnBatchNorm1d(nfm2)
        self.block2_conv2 = nnConv1d(nfm2, nfm2, kernel_size=5, padding="same")
        self.block2_bn2 = nnBatchNorm1d(nfm2)
        self.block2_conv3 = nnConv1d(nfm2, nfm2, kernel_size=3, padding="same")
        self.block2_bn3 = nnBatchNorm1d(nfm2)

        # Shortcut for block 2: project n_feature_maps -> n_feature_maps * 2
        self.block2_shortcut = nnSequential(
            nnConv1d(n_feature_maps, nfm2, kernel_size=1, padding="same"),
            nnBatchNorm1d(nfm2),
        )

        # ---- Residual Block 3 ----
        self.block3_conv1 = nnConv1d(nfm2, nfm2, kernel_size=8, padding="same")
        self.block3_bn1 = nnBatchNorm1d(nfm2)
        self.block3_conv2 = nnConv1d(nfm2, nfm2, kernel_size=5, padding="same")
        self.block3_bn2 = nnBatchNorm1d(nfm2)
        self.block3_conv3 = nnConv1d(nfm2, nfm2, kernel_size=3, padding="same")
        self.block3_bn3 = nnBatchNorm1d(nfm2)

        # Shortcut for block 3: no channel expansion, just BN
        self.block3_shortcut = nnBatchNorm1d(nfm2)

        # ---- Global Average Pooling ----
        self.gap = nnAdaptiveAvgPool1d(1)

        # ---- Output layer ----
        self.output_layer = nnLinear(nfm2, num_classes, bias=use_bias)

        # Build activation functions
        self._act_hidden = self._instantiate_activation(activation_hidden)
        self._act_output = None
        if activation_output is not None:
            self._act_output = self._instantiate_activation(activation_output)

    def forward(self, X):
        """Forward pass of the ResNet network.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``(batch_size, n_timesteps, n_dims)`` from
            the dataloader. Transposed internally to
            ``(batch_size, n_dims, n_timesteps)`` for Conv1d.

        Returns
        -------
        out : torch.Tensor
            Output tensor. Shape ``(batch_size, num_classes)`` for
            classification, ``(batch_size,)`` for regression.
        """
        # Dataloader provides (batch, n_timesteps, n_dims)
        # Conv1d expects (batch, n_dims, n_timesteps)
        if X.dim() == 3:
            X = X.transpose(1, 2)

        # ---- Block 1 ----
        shortcut = self.block1_shortcut(X)

        out = self.block1_conv1(X)
        out = self.block1_bn1(out)
        out = self._act_hidden(out)

        out = self.block1_conv2(out)
        out = self.block1_bn2(out)
        out = self._act_hidden(out)

        out = self.block1_conv3(out)
        out = self.block1_bn3(out)

        out = out + shortcut
        out = self._act_hidden(out)

        # ---- Block 2 ----
        shortcut = self.block2_shortcut(out)

        x2 = self.block2_conv1(out)
        x2 = self.block2_bn1(x2)
        x2 = self._act_hidden(x2)

        x2 = self.block2_conv2(x2)
        x2 = self.block2_bn2(x2)
        x2 = self._act_hidden(x2)

        x2 = self.block2_conv3(x2)
        x2 = self.block2_bn3(x2)

        x2 = x2 + shortcut
        out = self._act_hidden(x2)

        # ---- Block 3 ----
        shortcut = self.block3_shortcut(out)

        x3 = self.block3_conv1(out)
        x3 = self.block3_bn1(x3)
        x3 = self._act_hidden(x3)

        x3 = self.block3_conv2(x3)
        x3 = self.block3_bn2(x3)
        x3 = self._act_hidden(x3)

        x3 = self.block3_conv3(x3)
        x3 = self.block3_bn3(x3)

        x3 = x3 + shortcut
        out = self._act_hidden(x3)

        # ---- GAP ----
        out = self.gap(out)  # (batch, nfm2, 1)
        out = out.squeeze(-1)  # (batch, nfm2)

        # ---- Output layer ----
        out = self.output_layer(out)

        if self._act_output is not None:
            out = self._act_output(out)

        # regression: (batch, 1) -> (batch,)
        if self.num_classes == 1:
            return out.squeeze(-1)

        return out

    def _instantiate_activation(self, activation):
        """Instantiate a PyTorch activation function by name.

        Parameters
        ----------
        activation : str or torch.nn.Module
            Activation function name or instance.

        Returns
        -------
        activation_fn : torch.nn.Module
            The instantiated activation function.
        """
        if isinstance(activation, NNModule):
            return activation
        elif isinstance(activation, str):
            name = activation.lower()
            if name == "relu":
                return _safe_import("torch.nn.ReLU")()
            elif name == "sigmoid":
                return _safe_import("torch.nn.Sigmoid")()
            elif name == "softmax":
                return _safe_import("torch.nn.Softmax")(dim=1)
            elif name == "logsoftmax":
                return _safe_import("torch.nn.LogSoftmax")(dim=1)
            elif name == "logsigmoid":
                return _safe_import("torch.nn.LogSigmoid")()
            else:
                raise ValueError(
                    f"Unsupported activation '{activation}'. "
                    "Supported: 'relu', 'sigmoid', 'softmax', "
                    "'logsoftmax', 'logsigmoid'."
                )
        else:
            raise TypeError(
                f"`activation` must be str or torch.nn.Module, got {type(activation)}"
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict
        """
        params1 = {
            "n_channels": 1,
            "series_length": 20,
        }
        params2 = {
            "n_channels": 2,
            "series_length": 30,
            "n_feature_maps": 32,
            "random_state": 42,
        }
        return [params1, params2]
