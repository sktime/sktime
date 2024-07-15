"""Extra LTSF Model Layers."""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch.nn as nn

    nn_module = nn.Module
else:

    class nn_module:
        """Dummy class if torch is unavailable."""


class SeriesDecomposer:
    """Series decomposition block."""

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def _build(self):
        return self._SeriesDecomposer(self.kernel_size)

    class _SeriesDecomposer(nn_module):
        """Series decomposition block."""

        def __init__(self, kernel_size):
            super().__init__()
            self.moving_avg = MovingAverage(kernel_size, stride=1)._build()

        def forward(self, x):
            moving_mean = self.moving_avg(x)
            res = x - moving_mean
            return res, moving_mean


class MovingAverage:
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def _build(self):
        return self._MovingAverage(self.kernel_size, self.stride)

    class _MovingAverage(nn_module):
        """Moving average block to highlight the trend of time series."""

        def __init__(self, kernel_size, stride):
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        def forward(self, x):
            from torch import cat

            # padding on the both ends of time series
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            return x
