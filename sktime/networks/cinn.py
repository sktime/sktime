"""Conditional Invertible Neural Network (cINN) for forecasting."""

__author__ = ["benHeid"]

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn

    NNModule = nn.Module
else:

    class NNModule:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies("FrEIA", severity="none"):
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm


class CINNNetwork:
    """
    Conditional Invertible Neural Network.

    Parameters
    ----------
    horizon : int
        Forecasting horizon.
    cond_features : int
        Number of features in the condition.
    encoded_cond_size : int
        Dimension of the encoded condition.
    num_coupling_layers : int
        Number of coupling layers in the cINN.
    hidden_dim_size : int
        Number of hidden units in the subnet.
    activation : torch.nn.modules.Module
        Activation function to use in the subnet.
    """

    class _CINNNetwork(NNModule):
        def __init__(
            self,
            horizon,
            cond_features,
            encoded_cond_size=64,
            num_coupling_layers=15,
            hidden_dim_size=64,
            activation=None,
        ) -> None:
            super().__init__()
            self.cond_net = nn.Sequential(
                nn.Linear(cond_features * horizon, 128),
                nn.ReLU(),
                nn.Linear(128, encoded_cond_size),
            )
            self.hidden_dim_size = hidden_dim_size
            self.activation = activation if activation is not None else nn.ReLU
            self.network = self.build_inn(
                horizon, cond_features, encoded_cond_size, num_coupling_layers
            )
            self.trainable_parameters = [
                p for p in self.network.parameters() if p.requires_grad
            ]
            for p in self.trainable_parameters:
                p.data = 0.01 * torch.randn_like(p)
            if self.cond_net:
                self.trainable_parameters += list(self.cond_net.parameters())

        def build_inn(
            self, horizon, cond_features, encoded_cond_size, num_coupling_layers
        ):
            """
            Build the cINN.

            Parameters
            ----------
            horizon : int
                Forecasting horizon.
            cond_features : int
                Number of features in the condition.
            encoded_cond_size : int
                Dimension of the encoded condition.
            num_coupling_layers : int
                Number of coupling layers in the cINN.
            """
            nodes = [Ff.InputNode(horizon)]

            cond = Ff.ConditionNode(encoded_cond_size)

            for k in range(num_coupling_layers):
                nodes.append(
                    Ff.Node(
                        nodes[-1],
                        Fm.GLOWCouplingBlock,
                        {
                            "subnet_constructor": self.create_subnet(
                                hidden_dim_size=self.hidden_dim_size,
                                activation=self.activation,
                            )
                        },
                        conditions=cond,
                    )
                )
                nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}))
            return Ff.GraphINN(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

        def parameters(self, recurse: bool = True):
            """Return the trainable parameters of the cINN."""
            return self.trainable_parameters

        def create_subnet(self, hidden_dim_size, activation):
            """Create a subnet for the cINN.

            Parameters
            ----------
            hidden_dim_size : int, optional
                Number of hidden units in the subnet.
            activation : torch.nn.Module, optional
                Activation function to use in the subnet.
            """

            def get_subnet(ch_in, ch_out):
                return nn.Sequential(
                    nn.Linear(ch_in, hidden_dim_size),
                    activation(),
                    nn.Linear(hidden_dim_size, ch_out),
                )

            return get_subnet

        def forward(self, x, c, rev=False):
            """Forward pass through the cINN.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, horizon).
            c : torch.Tensor
                Condition tensor of shape (batch_size, cond_features * horizon).
            rev : bool, optional (default=False)
                Whether to run the reverse pass.
            """
            if isinstance(x, np.ndarray):
                if isinstance(c, np.ndarray):
                    c = self._calculate_condition(torch.from_numpy(c.astype("float32")))
                else:
                    c = self._calculate_condition(c)
                z, jac = self.network(
                    torch.from_numpy(x.astype("float32")), c=c, rev=rev
                )
            else:
                c = self._calculate_condition(c)
                z, jac = self.network(x.float(), c=c, rev=rev)
            return z, jac

        def _calculate_condition(self, c):
            if c is not None:
                c = self.cond_net(c.flatten(1))
            return c

        def reverse_sample(self, z, c):
            """
            Reverse sample from the cINN.

            Parameters
            ----------
            z : torch.Tensor
                Input tensor of shape (batch_size, horizon).
            c : torch.Tensor
                Condition tensor of shape (batch_size, cond_features * horizon).
            """
            c = self._calculate_condition(c)
            return self.network(z, c=c, rev=True)[0].detach().numpy()

    def __init__(
        self,
        horizon,
        cond_features,
        encoded_cond_size=64,
        num_coupling_layers=15,
        hidden_dim_size=64,
        activation=None,
    ) -> None:
        self.horizon = horizon
        self.cond_features = cond_features
        self.encoded_cond_size = encoded_cond_size
        self.num_coupling_layers = num_coupling_layers
        self.hidden_dim_size = hidden_dim_size
        self.activation = activation if activation is not None else nn.ReLU

    def build(self):
        """Build the cINN."""
        return self._CINNNetwork(
            self.horizon,
            self.cond_features,
            self.encoded_cond_size,
            self.num_coupling_layers,
            self.hidden_dim_size,
            self.activation,
        )
