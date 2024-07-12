"""Deep Learning Forecasters using LTSF-Linear Models."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


class LTSFLinearNetwork:
    """LTSF-Linear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) linear forecaster,
    aka LTSF-Linear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    in_channels : int, default=None
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear
    """

    class _LTSFLinearNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            pred_len,
            in_channels,
            individual,
        ):
            super().__init__()

            self.seq_len = seq_len
            self.pred_len = pred_len
            self.in_channels = in_channels
            self.individual = individual

            if self.individual:
                self.Linear = nn.ModuleList()
                for _ in range(self.in_channels):
                    self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
            else:
                self.Linear = nn.Linear(self.seq_len, self.pred_len)

        def forward(self, x):
            """Forward pass for LSTF-Linear Network.

            Parameters
            ----------
            x : torch.Tensor
                torch.Tensor of shape [Batch, Input Sequence Length, Channel]

            Returns
            -------
            x : torch.Tensor
                output of Linear Model. x.shape = [Batch, Output Length, Channel]
            """
            from torch import zeros

            if self.individual:
                output = zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(
                    x.device
                )
                for i in range(self.in_channels):
                    output[:, :, i] = self.Linear[i](x[:, :, i])
                x = output
            else:
                x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x  # [Batch, Output Length, Channel]

    def __init__(self, seq_len, pred_len, in_channels=1, individual=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.individual = individual

    def _build(self):
        return self._LTSFLinearNetwork(
            self.seq_len, self.pred_len, self.in_channels, self.individual
        )


class LTSFDLinearNetwork:
    """LTSF-DLinear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) decomposition linear
    forecaster, aka LTSF-DLinear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    in_channels : int, default=None
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear
    """

    class _LTSFDLinearNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            pred_len,
            in_channels,
            individual,
        ):
            from sktime.networks.ltsf.layers import SeriesDecomposer

            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len

            # Decompsition Kernel Size
            kernel_size = 25
            self.decompsition = SeriesDecomposer(kernel_size)._build()
            self.individual = individual
            self.in_channels = in_channels

            if self.individual:
                self.Linear_Seasonal = nn.ModuleList()
                self.Linear_Trend = nn.ModuleList()
                for _ in range(self.in_channels):
                    self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                    self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
            else:
                self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
                self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        def forward(self, x):
            """Forward pass for LSTF-DLinear Network.

            Parameters
            ----------
            x : torch.Tensor
                torch.Tensor of shape [Batch, Input Sequence Length, Channel]

            Returns
            -------
            x : torch.Tensor
                output of Linear Model. x.shape = [Batch, Output Length, Channel]
            """
            from torch import zeros

            # x: [Batch, Input length, Channel]
            seasonal_init, trend_init = self.decompsition(x)
            seasonal_init, trend_init = (
                seasonal_init.permute(0, 2, 1),
                trend_init.permute(0, 2, 1),
            )
            if self.individual:
                seasonal_output = zeros(
                    [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                    dtype=seasonal_init.dtype,
                ).to(seasonal_init.device)
                trend_output = zeros(
                    [trend_init.size(0), trend_init.size(1), self.pred_len],
                    dtype=trend_init.dtype,
                ).to(trend_init.device)
                for i in range(self.in_channels):
                    seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                        seasonal_init[:, i, :]
                    )
                    trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
            else:
                seasonal_output = self.Linear_Seasonal(seasonal_init)
                trend_output = self.Linear_Trend(trend_init)

            x = seasonal_output + trend_output
            return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

    def __init__(self, seq_len, pred_len, in_channels=1, individual=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.individual = individual

    def _build(self):
        return self._LTSFDLinearNetwork(
            self.seq_len, self.pred_len, self.in_channels, self.individual
        )


class LTSFNLinearNetwork:
    """LTSF-NLinear Forecaster.

    Implementation of the Long-Term Short-Term Feature (LTSF) normalization linear
    forecaster, aka LTSF-NLinear, by Zeng et al [1]_.

    Core logic is directly copied from the cure-lab LTSF-Linear implementation [2]_,
    which is unfortunately not available as a package.

    Parameters
    ----------
    seq_len : int
        length of input sequence
    pred_len : int
        length of prediction (forecast horizon)
    in_channels : int, default=None
        number of input channels passed to network
    individual : bool, default=False
        boolean flag that controls whether the network treats each channel individually"
        "or applies a single linear layer across all channels. If individual=True, the"
        "a separate linear layer is created for each input channel. If"
        "individual=False, a single shared linear layer is used for all channels."

    References
    ----------
    .. [1] Zeng A, Chen M, Zhang L, Xu Q. 2023.
    Are transformers effective for time series forecasting?
    Proceedings of the AAAI conference on artificial intelligence 2023
    (Vol. 37, No. 9, pp. 11121-11128).
    .. [2] https://github.com/cure-lab/LTSF-Linear
    """

    class _LTSFNLinearNetwork(nn_module):
        def __init__(
            self,
            seq_len,
            pred_len,
            in_channels,
            individual,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len

            self.in_channels = in_channels
            self.individual = individual

            if self.individual:
                self.Linear = nn.ModuleList()
                for _ in range(self.in_channels):
                    self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
            else:
                self.Linear = nn.Linear(self.seq_len, self.pred_len)

        def forward(self, x):
            """Forward pass for LSTF-NLinear Network.

            Parameters
            ----------
            x : torch.Tensor
                torch.Tensor of shape [Batch, Input Sequence Length, Channel]

            Returns
            -------
            x : torch.Tensor
                output of Linear Model. x.shape = [Batch, Output Length, Channel]
            """
            from torch import zeros

            # x: [Batch, Input length, Channel]
            seq_last = x[:, -1:, :].detach()
            x = x - seq_last
            if self.individual:
                output = zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(
                    x.device
                )
                for i in range(self.in_channels):
                    output[:, :, i] = self.Linear[i](x[:, :, i])
                x = output
            else:
                x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = x + seq_last
            return x  # [Batch, Output length, Channel]

    def __init__(self, seq_len, pred_len, in_channels=1, individual=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.individual = individual

    def _build(self):
        return self._LTSFNLinearNetwork(
            self.seq_len, self.pred_len, self.in_channels, self.individual
        )
