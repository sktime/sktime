from typing import Tuple

import torch
from torch import nn

from momentfm.utils.utils import control_randomness


class SyntheticDataset(nn.Module):
    def __init__(
        self,
        n_samples: int = 1024,
        seq_len: int = 512,
        freq: int = 1,
        freq_range: Tuple[int, int] = (1, 32),
        amplitude_range: Tuple[int, int] = (1, 32),
        trend_range: Tuple[int, int] = (1, 32),
        baseline_range: Tuple[int, int] = (1, 32),
        noise_mean: float = 0.0,
        noise_std: float = 0.1,
        random_seed: int = 42,
    ):
        super(SyntheticDataset, self).__init__()
        """
        Class to generate synthetic time series data.

        Parameters 
        ----------
        n_samples : int
            Number of samples to generate.
        seq_len : int
            Length of the time series.
        freq : int
            Frequency of the sinusoidal wave.
        freq_range : Tuple[int, int]
            Range of frequencies to generate.
        amplitude_range : Tuple[int, int]
            Range of amplitudes to generate.
        trend_range : Tuple[int, int]
            Range of trends to generate.
        baseline_range : Tuple[int, int]
            Range of baselines to generate.
        noise_mean : float
            Mean of the noise.
        noise_std : float
            Standard deviation of the noise.
        random_seed : int
            Random seed to control randomness.        
        """

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.freq = freq
        self.freq_range = freq_range
        self.amplitude_range = amplitude_range
        self.trend_range = trend_range
        self.baseline_range = baseline_range
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.random_seed = random_seed

        control_randomness(self.random_seed)

    def __repr__(self):
        return (
            f"SyntheticDataset(n_samples={self.n_samples},"
            + f"seq_len={self.seq_len},"
            + f"freq={self.freq},"
            + f"freq_range={self.freq_range},"
            + f"amplitude_range={self.amplitude_range},"
            + f"trend_range={self.trend_range},"
            + f"baseline_range={self.baseline_range},"
            + f"noise_mean={self.noise_mean},"
            + f"noise_std={self.noise_std},"
        )

    def _generate_noise(self):
        epsilon = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=(self.n_samples, self.seq_len),
        )

        return epsilon

    def _generate_x(self):
        t = (
            torch.linspace(start=0, end=1, steps=self.seq_len)
            .unsqueeze(0)
            .repeat(self.n_samples, 1)
        )
        x = 2 * self.freq * torch.pi * t
        return x, t

    def gen_sinusoids_with_varying_freq(self):
        c = (
            torch.linspace(
                start=self.freq_range[0], end=self.freq_range[1], steps=self.n_samples
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(c * x) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_correlation(self):
        c = (
            torch.linspace(start=0, end=2 * np.pi, steps=self.n_samples)
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x + c) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_amplitude(self):
        c = (
            torch.linspace(
                start=self.amplitude_range[0],
                end=self.amplitude_range[1],
                steps=self.n_samples,
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )

        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = c * torch.sin(x) + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_trend(self):
        c = (
            torch.linspace(
                start=self.trend_range[0], end=self.trend_range[1], steps=self.n_samples
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, t = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x) + t**c + epsilon
        y = y.unsqueeze(1)

        return y, c

    def gen_sinusoids_with_varying_baseline(self):
        c = (
            torch.linspace(
                start=self.baseline_range[0],
                end=self.baseline_range[1],
                steps=self.n_samples,
            )
            .unsqueeze(1)
            .repeat(1, self.seq_len)
        )
        x, _ = self._generate_x()
        epsilon = self._generate_noise()

        y = torch.sin(x) + c + epsilon
        y = y.unsqueeze(1)

        return y, c
