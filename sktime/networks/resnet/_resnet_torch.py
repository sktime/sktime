"""Residual Network (ResNet) for classification and regression in PyTorch."""

__authors__ = ["DCchoudhury15"]

from sktime.utils.dependencies import _safe_import

NNModule = _safe_import("torch.nn.Module")


class ResNetNetworkTorch(NNModule):
    """Establish the network structure for a ResNet in PyTorch.

    Parameters
    ----------
    input_size : int
        Number of input channels (number of time series dimensions).
    num_classes : int
        Number of output neurons. For classification, this is the number
        of classes. For regression, this is 1.
    n_feature_maps : int, default = 64
        Number of feature maps in the first residual block.
        Subsequent blocks use n_feature_maps * 2.
    activation_hidden : str, default = "relu"
        Activation function used in the residual blocks.
        Supported values are "relu", "tanh", "sigmoid".
    random_state : int, default = 0
        Seed to ensure reproducibility.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_feature_maps: int = 64,
        activation_hidden: str = "relu",
        random_state: int = 0,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.n_feature_maps = n_feature_maps
        self.activation_hidden = activation_hidden
        self.random_state = random_state

        super().__init__()

        Conv1d = _safe_import("torch.nn.Conv1d")
        BatchNorm1d = _safe_import("torch.nn.BatchNorm1d")
        Linear = _safe_import("torch.nn.Linear")

        # --- Block 1 ---
        self.conv1_x = Conv1d(
            input_size, self.n_feature_maps, kernel_size=8, padding="same"
        )
        self.bn1_x = BatchNorm1d(self.n_feature_maps)
        self.conv1_y = Conv1d(
            self.n_feature_maps, self.n_feature_maps, kernel_size=5, padding="same"
        )
        self.bn1_y = BatchNorm1d(self.n_feature_maps)
        self.conv1_z = Conv1d(
            self.n_feature_maps, self.n_feature_maps, kernel_size=3, padding="same"
        )
        self.bn1_z = BatchNorm1d(self.n_feature_maps)
        self.shortcut1 = Conv1d(
            input_size, self.n_feature_maps, kernel_size=1, padding="same"
        )
        self.shortcut1_bn = BatchNorm1d(self.n_feature_maps)

        # --- Block 2 ---
        self.conv2_x = Conv1d(
            self.n_feature_maps, self.n_feature_maps * 2, kernel_size=8, padding="same"
        )
        self.bn2_x = BatchNorm1d(self.n_feature_maps * 2)
        self.conv2_y = Conv1d(
            self.n_feature_maps * 2,
            self.n_feature_maps * 2,
            kernel_size=5,
            padding="same",
        )
        self.bn2_y = BatchNorm1d(self.n_feature_maps * 2)
        self.conv2_z = Conv1d(
            self.n_feature_maps * 2,
            self.n_feature_maps * 2,
            kernel_size=3,
            padding="same",
        )
        self.bn2_z = BatchNorm1d(self.n_feature_maps * 2)
        self.shortcut2 = Conv1d(
            self.n_feature_maps, self.n_feature_maps * 2, kernel_size=1, padding="same"
        )
        self.shortcut2_bn = BatchNorm1d(self.n_feature_maps * 2)

        # --- Block 3 ---
        self.conv3_x = Conv1d(
            self.n_feature_maps * 2,
            self.n_feature_maps * 2,
            kernel_size=8,
            padding="same",
        )
        self.bn3_x = BatchNorm1d(self.n_feature_maps * 2)
        self.conv3_y = Conv1d(
            self.n_feature_maps * 2,
            self.n_feature_maps * 2,
            kernel_size=5,
            padding="same",
        )
        self.bn3_y = BatchNorm1d(self.n_feature_maps * 2)
        self.conv3_z = Conv1d(
            self.n_feature_maps * 2,
            self.n_feature_maps * 2,
            kernel_size=3,
            padding="same",
        )
        self.bn3_z = BatchNorm1d(self.n_feature_maps * 2)
        self.shortcut3_bn = BatchNorm1d(self.n_feature_maps * 2)

        # --- Final layers ---
        _supported = {
            "relu": "torch.nn.ReLU",
            "tanh": "torch.nn.Tanh",
            "sigmoid": "torch.nn.Sigmoid",
        }
        if activation_hidden not in _supported:
            raise ValueError(
                f"activation_hidden must be one of {list(_supported.keys())}, "
                f"but got '{activation_hidden}'"
            )
        self.activation_fn = _safe_import(_supported[activation_hidden])()
        self.gap = _safe_import("torch.nn.AdaptiveAvgPool1d")(1)
        self.fc = Linear(self.n_feature_maps * 2, self.num_classes)

    def forward(self, X):
        """Forward pass through the ResNet network.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, series_length, input_size)
            Input tensor containing the time series data.

        Returns
        -------
        out : torch.Tensor of shape (batch_size, num_classes)
            Output tensor containing the logits for each class.
        """
        X = X.permute(0, 2, 1)

        # --- Block 1 ---
        out = self.activation_fn(self.bn1_x(self.conv1_x(X)))
        out = self.activation_fn(self.bn1_y(self.conv1_y(out)))
        out = self.bn1_z(self.conv1_z(out))
        shortcut = self.shortcut1_bn(self.shortcut1(X))
        out = self.activation_fn(out + shortcut)

        # --- Block 2 ---
        out2 = self.activation_fn(self.bn2_x(self.conv2_x(out)))
        out2 = self.activation_fn(self.bn2_y(self.conv2_y(out2)))
        out2 = self.bn2_z(self.conv2_z(out2))
        shortcut2 = self.shortcut2_bn(self.shortcut2(out))
        out2 = self.activation_fn(out2 + shortcut2)

        # --- Block 3 ---
        out3 = self.activation_fn(self.bn3_x(self.conv3_x(out2)))
        out3 = self.activation_fn(self.bn3_y(self.conv3_y(out3)))
        out3 = self.bn3_z(self.conv3_z(out3))
        shortcut3 = self.shortcut3_bn(out2)
        out3 = self.activation_fn(out3 + shortcut3)

        # --- GAP + final layer ---
        out3 = self.gap(out3).squeeze(-1)
        return self.fc(out3)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {
            "input_size": 1,
            "num_classes": 2,
        }
        params2 = {
            "input_size": 2,
            "num_classes": 3,
            "n_feature_maps": 32,
            "random_state": 42,
        }
        return [params1, params2]
