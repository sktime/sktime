"""Residual Network (ResNet) for classification and regression in PyTorch."""

__authors__ = ["amitsubhashchejara"]


import numpy as np

# from sktime.networks.base import BaseDeepNetwork
from sktime.utils.dependencies import _safe_import

# handling soft dependencies for Torch modules
Torch = _safe_import("torch")
NNModule = _safe_import("torch.nn.Module")
F = _safe_import("torch.nn.functional")
Conv2d = _safe_import("torch.nn.Conv2d")
BatchNorm2d = _safe_import("torch.nn.BatchNorm2d")


class ResNetNetworkTorch(NNModule):
    """Establish the network structure for a ResNet in PyTorch.

    Adapted from the implementation in
    https://github.com/aybchan/time-series-classification/blob/master/src/project/ResNet.py

    Parameters
    ----------
    input_size : int or tuple
        If int, it represents the number of expected features in the input.
        If tuple, it must be of the form (n_instances, n_dims, series_length)
        where n_dims is the number of expected features in the input.
    num_classes : int
        Number of classes to predict.
    random_state : int, optional (default = 0)
        The random seed to use random activities.
    activation : string, optional (default = "relu")
        Activation function used for hidden layers;
        List of available torch activation functions:
        https://pytorch.org/docs/stable/nn.functional.html#activation-functions

    References
    ----------
    .. [1] H. Fawaz, G. B. Lanckriet, F. Petitjean, and L. Idoumghar,

    Network originally defined in:

    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["amitsubhashchejara"],
        "python_version": ">=3.9",
        "python_dependencies": "torch",
        "property:randomness": "stochastic",
        "capability:random_state": True,
    }

    def __init__(
        self,
        input_size: int | tuple,
        num_classes: int,
        random_state: int = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.random_state = random_state
        self.activation = activation

        if self.random_state is not None:
            Torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.blocks = NNModule.ModuleList()

        # Checking input dimensions.
        if isinstance(input_size, int):
            in_features = input_size
        elif isinstance(input_size, tuple):
            if len(input_size) == 3:
                in_features = input_size[1]
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

        blocks = [in_features, 64, 128, 128]

        for b, _ in enumerate(blocks[:-1]):
            self.blocks.append(
                ResidualBlock(
                    *blocks[b : b + 2], in_features, activation=self.activation
                )
            )

        self.fc1 = NNModule.Linear(blocks[-1], self.num_classes)

    def forward(self, X):
        for block in self.blocks:
            X = block(X)

        X = F.avg_pool2d(X, 2)
        X = Torch.mean(X, dim=2)
        X = X.view(-1, 1, 128)
        X = self.fc1(X)

        return X.view(-1, self.num_classes)


class ResidualBlock(NNModule):
    def __init__(self, in_maps, out_maps, time_steps, activation="relu"):
        super().__init__()
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.time_steps = time_steps
        self.activation = getattr(F, activation)

        self.conv1 = Conv2d(self.in_maps, self.out_maps, (7, 1), 1, (3, 0))
        self.bn1 = BatchNorm2d(self.out_maps)

        self.conv2 = Conv2d(self.out_maps, self.out_maps, (5, 1), 1, (2, 0))
        self.bn2 = BatchNorm2d(self.out_maps)

        self.conv3 = Conv2d(self.out_maps, self.out_maps, (3, 1), 1, (1, 0))
        self.bn3 = BatchNorm2d(self.out_maps)

    def forward(self, X):
        if isinstance(X, np.ndarray):
            torchFrom_numpy = _safe_import("torch.from_numpy")
            X = torchFrom_numpy(X).float()

        X = X.view(-1, self.in_maps, self.time_steps, 1)
        X = self.activation(self.bn1(self.conv1(X)))
        inx = X
        X = self.activation(self.bn2(self.conv2(X)))
        X = self.activation(self.bn3(self.conv3(X)) + inx)

        return X
