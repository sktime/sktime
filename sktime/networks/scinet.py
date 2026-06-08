"""Deep Learning Forecasters using SCINet Forecaster."""

import math

import numpy as np

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


class Splitting:
    """
    A neural network module for splitting input into even and odd parts.

    The `_Splitting` class defines a PyTorch module that splits an input tensor into
    two parts: the even-indexed and odd-indexed elements. This is useful in tasks
    requiring input splitting or masking.

    This class wraps the `Splitting` neural network module to provide an additional
    `_build` method for instantiating the wrapped `Splitting` class. The `Splitting`
    module is used for splitting an input tensor into even-indexed and odd-indexed
    parts.

    Methods
    -------
    _build :
        Creates an instance of the wrapped `Splitting` class.
    """

    class _Splitting(nn_module):
        def __init__(self):
            super().__init__()

        def even(self, x):
            """
            Extract even-indexed elements from the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape [Batch, Sequence Length, Channels].

            Returns
            -------
            torch.Tensor
                Tensor containing even-indexed elements of `x`.
            """
            return x[:, ::2, :]

        def odd(self, x):
            """
            Extract odd-indexed elements from the input tensor.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape [Batch, Sequence Length, Channels].

            Returns
            -------
            torch.Tensor
                Tensor containing odd-indexed elements of `x`.
            """
            return x[:, 1::2, :]

        def forward(self, x):
            """
            Split the input tensor into even and odd parts.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape [Batch, Sequence Length, Channels].

            Returns
            -------
            tuple
                A tuple containing:
                - Even-indexed elements as a tensor.
                - Odd-indexed elements as a tensor.
            """
            return (self.even(x), self.odd(x))

    def __init__(self):
        """Initialize the SplittingModule wrapper."""
        pass

    def _build(self):
        """Instantiate the wrapped `_Splitting` class."""
        return self._Splitting()


class Interactor:
    """
    Neural network module for tensor operations.

    This class wraps the Interactor module, providing a convenient
    interface for its initialization with predefined parameters.

    Parameters
    ----------
    in_planes (int): Number of input planes (channels) in the input tensor.
    splitting (bool, optional): Whether to split the input tensor. Defaults to True.
    kernel (int, optional): The kernel size for convolution layers. Defaults to 5.
    dropout (float, optional): The dropout rate for regularization. Defaults to 0.5.
    groups (int, optional): Number of groups for convolutions. Defaults to 1.
    hidden_size (int, optional): Size of hidden layer. Defaults to 1.
    INN (bool, optional): Whether to apply modifications to the network.
    Defaults to True.
    """

    class _Interactor(nn_module):
        def __init__(
            self,
            in_planes,
            splitting=True,
            kernel=5,
            dropout=0.5,
            groups=1,
            hidden_size=1,
            INN=True,
        ):
            super().__init__()
            self.modified = INN
            self.kernel_size = kernel
            self.dilation = 1
            self.dropout = dropout
            self.hidden_size = hidden_size
            self.groups = groups

            if self.kernel_size % 2 == 0:
                pad_l = (
                    self.dilation * (self.kernel_size - 2) // 2 + 1
                )  # by default: stride==1
                pad_r = (
                    self.dilation * (self.kernel_size) // 2 + 1
                )  # by default: stride==1
            else:
                pad_l = (
                    self.dilation * (self.kernel_size - 1) // 2 + 1
                )  # we fix the kernel size of the second layer as 3.
                pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
            self.splitting = splitting
            self.split = Splitting()._build()

            modules_P = []
            modules_U = []
            modules_psi = []
            modules_phi = []
            prev_size = 1

            size_hidden = self.hidden_size
            modules_P += [
                nn.ReplicationPad1d((pad_l, pad_r)),
                nn.Conv1d(
                    in_planes * prev_size,
                    int(in_planes * size_hidden),
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    stride=1,
                    groups=self.groups,
                ),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv1d(
                    int(in_planes * size_hidden),
                    in_planes,
                    kernel_size=3,
                    stride=1,
                    groups=self.groups,
                ),
                nn.Tanh(),
            ]
            modules_U += [
                nn.ReplicationPad1d((pad_l, pad_r)),
                nn.Conv1d(
                    in_planes * prev_size,
                    int(in_planes * size_hidden),
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    stride=1,
                    groups=self.groups,
                ),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv1d(
                    int(in_planes * size_hidden),
                    in_planes,
                    kernel_size=3,
                    stride=1,
                    groups=self.groups,
                ),
                nn.Tanh(),
            ]

            modules_phi += [
                nn.ReplicationPad1d((pad_l, pad_r)),
                nn.Conv1d(
                    in_planes * prev_size,
                    int(in_planes * size_hidden),
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    stride=1,
                    groups=self.groups,
                ),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv1d(
                    int(in_planes * size_hidden),
                    in_planes,
                    kernel_size=3,
                    stride=1,
                    groups=self.groups,
                ),
                nn.Tanh(),
            ]
            modules_psi += [
                nn.ReplicationPad1d((pad_l, pad_r)),
                nn.Conv1d(
                    in_planes * prev_size,
                    int(in_planes * size_hidden),
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    stride=1,
                    groups=self.groups,
                ),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv1d(
                    int(in_planes * size_hidden),
                    in_planes,
                    kernel_size=3,
                    stride=1,
                    groups=self.groups,
                ),
                nn.Tanh(),
            ]
            self.phi = nn.Sequential(*modules_phi)
            self.psi = nn.Sequential(*modules_psi)
            self.P = nn.Sequential(*modules_P)
            self.U = nn.Sequential(*modules_U)

        def forward(self, x):
            """
            Forward pass of the Interactor module.

            Splits the input if 'splitting' is True, and applies the
            respective operations. If 'modified' is True, applies
            modified equations, otherwise applies the default operations.

            x (Tensor): The input tensor to be processed by the module.

            Returns
            -------
            tuple: Two tensors (x_even_update, x_odd_update) or (c, d)
            """
            if self.splitting:
                (x_even, x_odd) = self.split(x)
            else:
                (x_even, x_odd) = x

            if self.modified:
                x_even = x_even.permute(0, 2, 1)
                x_odd = x_odd.permute(0, 2, 1)

                d = x_odd.mul(torch.exp(self.phi(x_even)))
                c = x_even.mul(torch.exp(self.psi(x_odd)))

                x_even_update = c + self.U(d)
                x_odd_update = d - self.P(c)

                return (x_even_update, x_odd_update)

            else:
                x_even = x_even.permute(0, 2, 1)
                x_odd = x_odd.permute(0, 2, 1)

                d = x_odd - self.P(x_even)
                c = x_even + self.U(d)

                return (c, d)

    def __init__(
        self,
        in_planes,
        splitting=True,
        kernel=5,
        dropout=0.5,
        groups=1,
        hidden_size=1,
        INN=True,
    ):
        """Initialize the Interactor module wrapper."""
        # Initialize all class parameters with provided arguments or default values
        self.in_planes = in_planes
        self.splitting = splitting
        self.kernel = kernel
        self.dropout = dropout
        self.groups = groups
        self.hidden_size = hidden_size
        self.INN = INN

    def _build(self):
        """Instantiate the wrapped Interactor module with predefined parameters.

        Returns
        -------
        Interactor : Interactor
            The initialized Interactor module with default parameters.
        """
        return self._Interactor(
            in_planes=self.in_planes,
            splitting=self.splitting,
            kernel=self.kernel,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=self.INN,
        )


class InteractorLevel:
    """InteractorLevel class for applying Interactor at a specific level.

    Parameters
    ----------
    in_planes (int): Number of input channels.
    kernel (int): Kernel size for the convolution layer.
    dropout (float): Dropout rate for regularization.
    groups (int): Number of groups for group convolutions.
    hidden_size (int): Size of the hidden layers in the network.
    INN (bool): A flag to enable/disable the INN architecture.
    """

    class _InteractorLevel(nn_module):
        def __init__(self, in_planes, kernel, dropout, groups, hidden_size, INN):
            """Initialize the InteractorLevel module."""
            super().__init__()
            self.level = Interactor(
                in_planes=in_planes,
                splitting=True,
                kernel=kernel,
                dropout=dropout,
                groups=groups,
                hidden_size=hidden_size,
                INN=INN,
            )._build()

        def forward(self, x):
            """
            Forward pass for the InteractorLevel.

            The forward method takes an input tensor `x`, passes it through
            the Interactor, and returns the updated even and odd parts of
            the tensor. These parts are the result of the interaction
            operation at the specified level.
            """
            (x_even_update, x_odd_update) = self.level(x)
            return (x_even_update, x_odd_update)

    def __init__(self, in_planes, kernel, dropout, groups, hidden_size, INN):
        """Initialize the wrapper of the InteractorLevel module."""
        self.in_planes = in_planes
        self.kernel = kernel
        self.dropout = dropout
        self.groups = groups
        self.hidden_size = hidden_size
        self.INN = INN

    def _build(self):
        """Instantiate the InteractorLevel module with the specified parameters."""
        return self._InteractorLevel(
            in_planes=self.in_planes,
            kernel=self.kernel,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=self.INN,
        )


class LevelSCINet:
    """
    A neural network class for LevelSCINet with interaction layers.

    Parameters
    ----------
        in_planes (int): Number of input channels.
        kernel_size (int): Size of the convolutional kernel.
        dropout (float): Dropout rate to avoid overfitting.
        groups (int): Number of groups for grouped convolutions.
        hidden_size (int): Hidden layer size for the interaction network.
        INN (bool): Whether to use an invertible neural network.
    """

    class _LevelSCINet(nn_module):
        def __init__(self, in_planes, kernel_size, dropout, groups, hidden_size, INN):
            """Initialize the LevelSCINet with an InteractorLevel module."""
            super().__init__()
            self.interact = InteractorLevel(
                in_planes=in_planes,
                kernel=kernel_size,
                dropout=dropout,
                groups=groups,
                hidden_size=hidden_size,
                INN=INN,
            )._build()

        def forward(self, x):
            """
            Forward pass through the LevelSCINet.

            Parameters
            ----------
                x (tensor): The input tensor to the network.

            Returns
            -------
                tuple: A tuple containing the even and odd updates after
                interaction, with dimensions (B, T, D) for both even and odd.
            """
            (x_even_update, x_odd_update) = self.interact(x)
            return x_even_update.permute(0, 2, 1), x_odd_update.permute(
                0, 2, 1
            )  # even: B, T, D odd: B, T, D

    def __init__(self, in_planes, kernel_size, dropout, groups, hidden_size, INN):
        """Initialize the LevelSCINet wrapper module."""
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.groups = groups
        self.hidden_size = hidden_size
        self.INN = INN

    def _build(self):
        """Build LevelSCINet actual module."""
        return self._LevelSCINet(
            in_planes=self.in_planes,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=self.INN,
        )


class SCINet_Tree:
    """SCINet_Tree class implements a recursive SCINet model.

    Parameters
    ----------
    in_planes: The number of input planes (features) for the model.
    current_level: The current level in the recursive SCINet tree.
    kernel_size: The kernel size used in convolution operations.
    dropout: Dropout rate for regularization.
    groups: Number of groups for group convolutions.
    hidden_size: Size of the hidden layer in SCINet.
    INN: A boolean flag for using the INN module.
    """

    class _SCINet_Tree(nn_module):
        def __init__(
            self,
            in_planes,
            current_level,
            kernel_size,
            dropout,
            groups,
            hidden_size,
            INN,
        ):
            """
            Initialize the SCINet_Tree module and its sub-modules.

            Sets the current_level and initializes the working block as
            a LevelSCINet. If not at the base level, it recursively creates
            sub-modules for even and odd levels of SCINet_Tree.
            """
            super().__init__()
            self.current_level = current_level

            self.workingblock = LevelSCINet(
                in_planes=in_planes,
                kernel_size=kernel_size,
                dropout=dropout,
                groups=groups,
                hidden_size=hidden_size,
                INN=INN,
            )._build()

            if current_level != 0:
                self.SCINet_Tree_odd = SCINet_Tree(
                    in_planes,
                    current_level - 1,
                    kernel_size,
                    dropout,
                    groups,
                    hidden_size,
                    INN,
                )._build()
                self.SCINet_Tree_even = SCINet_Tree(
                    in_planes,
                    current_level - 1,
                    kernel_size,
                    dropout,
                    groups,
                    hidden_size,
                    INN,
                )._build()

        def zip_up_the_pants(self, even, odd):
            """
            Zips up the even and odd indexed sub-sequences.

            The method permutes and interleaves the even and odd subsequences
            from input tensors and ensures that they have the same length by
            padding the shorter one, then returns the interleaved result.
            """
            even = even.permute(1, 0, 2)
            odd = odd.permute(1, 0, 2)  # L, B, D
            even_len = even.shape[0]
            odd_len = odd.shape[0]
            mlen = min((odd_len, even_len))
            _ = []
            for i in range(mlen):
                _.append(even[i].unsqueeze(0))
                _.append(odd[i].unsqueeze(0))
            if odd_len < even_len:
                _.append(even[-1].unsqueeze(0))
            return torch.cat(_, 0).permute(1, 0, 2)  # B, L, D

        def forward(self, x):
            """
            Forward pass of the SCINet_Tree model.

            The input is processed through the working block, and depending
            on the current_level, the output is recursively passed through
            even and odd sub-trees. If at the base level, the zipped output
            is returned after permutation.
            """
            x_even_update, x_odd_update = self.workingblock(x)
            # We recursively reordered these sub-series. You can run the
            # ./utils/recursive_demo.py to emulate this procedure.
            if self.current_level == 0:
                return self.zip_up_the_pants(x_even_update, x_odd_update)
            else:
                return self.zip_up_the_pants(
                    self.SCINet_Tree_even(x_even_update),
                    self.SCINet_Tree_odd(x_odd_update),
                )

    def __init__(
        self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN
    ):
        """Initialize the wrapper of SCINet tree module."""
        # Adjusted values based on the input parameters
        self.in_planes = in_planes
        self.current_level = current_level
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.groups = groups
        self.hidden_size = hidden_size
        self.INN = INN

    def _build(self):
        """Build actual SCINeT_Tree module."""
        return self._SCINet_Tree(
            in_planes=self.in_planes,
            current_level=self.current_level,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=self.INN,
        )


class EncoderTree:
    """EncoderTree encodes data through SCINet levels.

    Parameters
    ----------
    in_planes (int): The number of input channels in the input data.
    num_levels (int): The number of hierarchical levels in the network.
    kernel_size (int): The size of the convolutional kernel.
    dropout (float): The dropout rate to prevent overfitting.
    groups (int): The number of groups for group convolutions.
    hidden_size (int): The size of the hidden layer.
    INN (bool): A flag indicating whether to use an INN network.
    """

    class _EncoderTree(nn_module):
        def __init__(
            self, in_planes, num_levels, kernel_size, dropout, groups, hidden_size, INN
        ):
            """Initialize the EncoderTree class and the SCINet_Tree instance."""
            super().__init__()
            self.levels = num_levels
            self.SCINet_Tree = SCINet_Tree(
                in_planes=in_planes,
                current_level=num_levels - 1,
                kernel_size=kernel_size,
                dropout=dropout,
                groups=groups,
                hidden_size=hidden_size,
                INN=INN,
            )._build()

        def forward(self, x):
            """
            Forward pass for the EncoderTree model.

            x (Tensor): Input tensor passed through the SCINet_Tree.

            Returns
            -------
            Tensor: Output tensor from the SCINet_Tree after processing.
            """
            x = self.SCINet_Tree(x)

            return x

    def __init__(
        self, in_planes, num_levels, kernel_size, dropout, groups, hidden_size, INN
    ):
        """Initialize Encoder Tree wrapper class."""
        self.in_planes = in_planes
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.groups = groups
        self.hidden_size = hidden_size
        self.INN = INN

    def _build(self):
        """Build encoder tree actual module."""
        return self._EncoderTree(
            in_planes=self.in_planes,
            num_levels=self.num_levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=self.INN,
        )


class SCINet:
    """SCINet: Time-Series Forecasting Model.

    Implementation of the SCINet model for time series forecasting, leveraging
    a stack-based encoder-decoder architecture to capture temporal dependencies
    effectively.

    SCINet supports features such as positional encoding, reverse instance
    normalization (RIN), and flexible decoder layers to enhance forecasting accuracy.

    Parameters
    ----------
    seq_len : int
        Length of the input sequence.
    pred_len : int
        Length of the prediction (forecast horizon).
    input_dim : int, default=9
        Number of input features (dimensions) per time step.
    hid_size : int, default=1
        Size of the hidden layer in the encoder.
    num_stacks : int, default=1
        Number of stack-based encoder-decoder layers. Maximum of two stacks
        supported.
    num_levels : int, default=3
        Number of hierarchical levels in the encoder tree.
    num_decoder_layer : int, default=1
        Number of layers in the decoder for producing predictions.
    concat_len : int, default=0
        Length of concatenation for intermediate outputs in multi-stack models.
    groups : int, default=1
        Number of groups for grouped convolutions in encoder levels.
    kernel : int, default=5
        Size of the convolutional kernel.
    dropout : float, default=0.5
        Dropout rate to prevent overfitting.
    single_step_output_One : int, default=0
        Flag to enable single-step output.
    positionalE : bool, default=False
        Whether to include positional encoding in the input sequence.
    modified : bool, default=True
        Flag to enable modifications in the SCINet encoder.
    RIN : bool, default=False
        Flag to enable Reverse Instance Normalization (RIN) for input
        normalization.
    """

    class _SCINet(nn_module):
        def __init__(
            self,
            pred_len,
            seq_len,
            input_dim=9,
            hid_size=1,
            num_stacks=1,
            num_levels=3,
            num_decoder_layer=1,
            concat_len=0,
            groups=1,
            kernel=5,
            dropout=0.5,
            single_step_output_One=0,
            positionalE=False,
            modified=True,
            RIN=False,
        ):
            super().__init__()

            self.input_dim = input_dim
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.hidden_size = hid_size
            self.num_levels = num_levels
            self.groups = groups
            self.modified = modified
            self.kernel_size = kernel
            self.dropout = dropout
            self.single_step_output_One = single_step_output_One
            self.concat_len = concat_len
            self.pe = positionalE
            self.RIN = RIN
            self.num_decoder_layer = num_decoder_layer

            self.blocks1 = EncoderTree(
                in_planes=self.input_dim,
                num_levels=self.num_levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                groups=self.groups,
                hidden_size=self.hidden_size,
                INN=modified,
            )._build()

            if num_stacks == 2:  # we only implement two stacks at most.
                self.blocks2 = EncoderTree(
                    in_planes=self.input_dim,
                    num_levels=self.num_levels,
                    kernel_size=self.kernel_size,
                    dropout=self.dropout,
                    groups=self.groups,
                    hidden_size=self.hidden_size,
                    INN=modified,
                )._build()

            self.stacks = num_stacks

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
            self.projection1 = nn.Conv1d(
                self.seq_len, self.pred_len, kernel_size=1, stride=1, bias=False
            )
            self.div_projection = nn.ModuleList()
            self.overlap_len = self.seq_len // 4
            self.div_len = self.seq_len // 6

            if self.num_decoder_layer > 1:
                self.projection1 = nn.Linear(self.seq_len, self.pred_len)
                for layer_idx in range(self.num_decoder_layer - 1):
                    div_projection = nn.ModuleList()
                    for i in range(6):
                        lens = (
                            min(i * self.div_len + self.overlap_len, self.seq_len)
                            - i * self.div_len
                        )
                        div_projection.append(nn.Linear(lens, self.div_len))
                    self.div_projection.append(div_projection)

            if self.single_step_output_One:  # only output the N_th timestep.
                if self.stacks == 2:
                    if self.concat_len:
                        self.projection2 = nn.Conv1d(
                            self.concat_len + self.pred_len,
                            1,
                            kernel_size=1,
                            bias=False,
                        )
                    else:
                        self.projection2 = nn.Conv1d(
                            self.seq_len + self.pred_len, 1, kernel_size=1, bias=False
                        )
            else:  # output the N timesteps.
                if self.stacks == 2:
                    if self.concat_len:
                        self.projection2 = nn.Conv1d(
                            self.concat_len + self.pred_len,
                            self.pred_len,
                            kernel_size=1,
                            bias=False,
                        )
                    else:
                        self.projection2 = nn.Conv1d(
                            self.seq_len + self.pred_len,
                            self.pred_len,
                            kernel_size=1,
                            bias=False,
                        )

            # For positional encoding
            self.pe_hidden_size = input_dim
            if self.pe_hidden_size % 2 == 1:
                self.pe_hidden_size += 1

            num_timescales = self.pe_hidden_size // 2
            max_timescale = 10000.0
            min_timescale = 1.0

            log_timescale_increment = math.log(
                float(max_timescale) / float(min_timescale)
            ) / max(num_timescales - 1, 1)
            # temp = torch.arange(num_timescales, dtype=torch.float32)
            inv_timescales = min_timescale * torch.exp(
                torch.arange(num_timescales, dtype=torch.float32)
                * -log_timescale_increment
            )
            self.register_buffer("inv_timescales", inv_timescales)

            ### RIN Parameters ###
            if self.RIN:
                self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
                self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))

        def get_position_encoding(self, x):
            """Generate position encoding for the input tensor `x`."""
            max_length = x.size()[1]
            position = torch.arange(
                max_length, dtype=torch.float32, device=x.device
            )  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
            # temp1 = position.unsqueeze(1)  # 5 1
            # temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
            scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(
                0
            )  # 5 256
            signal = torch.cat(
                [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
            )  # [T, C]
            signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
            signal = signal.view(1, max_length, self.pe_hidden_size)

            return signal

        def forward(self, x):
            """Perform the forward pass through the network.

            Ensure that the sequence length (`self.seq_len`) is divisible by 2 raised
            to the power of the number of levels (`self.num_levels`).

            Raises
            ------
            AssertionError
                If `seq_len` is not divisible by 2^num_levels`.
            """
            assert self.seq_len % (np.power(2, self.num_levels)) == 0
            # evenly divided the input length into two parts.
            # (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
            if self.pe:
                pe = self.get_position_encoding(x)
                if pe.shape[2] > x.shape[2]:
                    x += pe[:, :, :-1]
                else:
                    x += self.get_position_encoding(x)

            ### activated when RIN flag is set ###
            if self.RIN:
                print("/// RIN ACTIVATED ///\r", end="")
                means = x.mean(1, keepdim=True).detach()
                # mean
                x = x - means
                # var
                stdev = torch.sqrt(
                    torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
                )
                x /= stdev
                # affine
                # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
                x = x * self.affine_weight + self.affine_bias

            # the first stack
            res1 = x
            x = self.blocks1(x)
            x += res1
            if self.num_decoder_layer == 1:
                x = self.projection1(x)
            else:
                x = x.permute(0, 2, 1)
                for div_projection in self.div_projection:
                    output = torch.zeros(x.shape, dtype=x.dtype).cuda()
                    for i, div_layer in enumerate(div_projection):
                        div_x = x[
                            :,
                            :,
                            i * self.div_len : min(
                                i * self.div_len + self.overlap_len, self.seq_len
                            ),
                        ]
                        output[:, :, i * self.div_len : (i + 1) * self.div_len] = (
                            div_layer(div_x)
                        )
                    x = output
                x = self.projection1(x)
                x = x.permute(0, 2, 1)

            if self.stacks == 1:
                ### reverse RIN ###
                if self.RIN:
                    x = x - self.affine_bias
                    x = x / (self.affine_weight + 1e-10)
                    x = x * stdev
                    x = x + means

                return x

            elif self.stacks == 2:
                MidOutPut = x
                if self.concat_len:
                    x = torch.cat((res1[:, -self.concat_len :, :], x), dim=1)
                else:
                    x = torch.cat((res1, x), dim=1)

                # the second stack
                res2 = x
                x = self.blocks2(x)
                x += res2
                x = self.projection2(x)

                ### Reverse RIN ###
                if self.RIN:
                    MidOutPut = MidOutPut - self.affine_bias
                    MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                    MidOutPut = MidOutPut * stdev
                    MidOutPut = MidOutPut + means

                if self.RIN:
                    x = x - self.affine_bias
                    x = x / (self.affine_weight + 1e-10)
                    x = x * stdev
                    x = x + means

                return x, MidOutPut

    def __init__(
        self,
        pred_len,
        seq_len,
        input_dim=9,
        hid_size=1,
        num_stacks=1,
        num_levels=3,
        num_decoder_layer=1,
        concat_len=0,
        groups=1,
        kernel=5,
        dropout=0.5,
        single_step_output_One=0,
        positionalE=False,
        modified=True,
        RIN=False,
    ):
        """Initialize SCINet wrapper."""
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hid_size = hid_size
        self.num_stacks = num_stacks
        self.num_levels = num_levels
        self.num_decoder_layer = num_decoder_layer
        self.concat_len = concat_len
        self.groups = groups
        self.kernel = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.positionalE = positionalE
        self.modified = modified
        self.RIN = RIN

    def _build(self):
        """Initialize actual SCINet model from self._SCINet."""
        return self._SCINet(
            pred_len=self.pred_len,
            seq_len=self.seq_len,
            input_dim=self.input_dim,
            hid_size=self.hid_size,
            num_stacks=self.num_stacks,
            num_levels=self.num_levels,
            num_decoder_layer=self.num_decoder_layer,
            concat_len=self.concat_len,
            groups=self.groups,
            kernel=self.kernel,
            dropout=self.dropout,
            single_step_output_One=self.single_step_output_One,
            positionalE=self.positionalE,
            modified=self.modified,
            RIN=self.RIN,
        )


def get_variable(x):
    """Convert the input to a Variable and move it to CUDA if available."""
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
