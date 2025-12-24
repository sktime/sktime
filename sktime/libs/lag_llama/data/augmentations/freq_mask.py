# Copyright 2024 Arjun Ashok
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch"):
    import torch


@torch.no_grad()
def freq_mask(x, y, rate=0.1, dim=1):
    # Get lengths of the input tensors along the specified dimension.
    x_len = x.shape[dim]
    y_len = y.shape[dim]

    # Concatenate x and y along the specified dimension.
    # x and y represent past and future targets respectively.
    xy = torch.cat([x, y], dim=dim)

    # Perform a real-valued fast Fourier transform (RFFT) on the concatenated tensor.
    # This transforms the time series data into the frequency domain.
    xy_f = torch.fft.rfft(xy, dim=dim)

    # Create a random mask with a probability defined by 'rate'.
    # This mask will be used to randomly select frequencies to be zeroed out.
    m = torch.rand_like(xy_f, dtype=xy.dtype) < rate

    # Apply the mask to the real and imaginary parts of the frequency data,
    # setting the selected frequencies to zero. This 'masks' those frequencies.
    freal = xy_f.real.masked_fill(m, 0)
    fimag = xy_f.imag.masked_fill(m, 0)

    # Combine the masked real and imaginary parts back into complex frequency data.
    xy_f = torch.complex(freal, fimag)

    # Perform an inverse RFFT to transform the data back to the time domain.
    # The masked frequencies will affect the reconstructed time series.
    xy = torch.fft.irfft(xy_f, dim=dim)

    # If the reconstructed data length differs from the original concatenated length,
    # adjust it to maintain consistency. This step ensures the output shape matches the input.
    if x_len + y_len != xy.shape[dim]:
        xy = torch.cat([x[:, 0:1, ...], xy], 1)

    # Split the reconstructed data back into two parts corresponding to the original x and y.
    return torch.split(xy, [x_len, y_len], dim=dim)
