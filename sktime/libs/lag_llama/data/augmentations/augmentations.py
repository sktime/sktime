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

# Adapted from https://github.com/vafl/gluon-ts/blob/ts_embeddings/src/gluonts/nursery/ts_embeddings/pt_augmentation.py

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline


class ApplyAugmentations(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transformation = RandomApply(transforms)

    def forward(self, ip1, ip2):
        ip_concat = torch.concat((ip1, ip2), dim=1)
        ip_aug = self.transformation(ip_concat)
        op1, op2 = torch.split(ip_aug, [ip1.size(1), ip2.size(1)], dim=1)
        assert len(op1.shape) == 2 and len(op2.shape) == 2
        return op1, op2


class RandomApply(nn.Module):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        # 'transforms' is a list of transformation modules to be applied to the input data.
        # These could be any transformations like normalization, augmentation, etc.
        self.transforms = nn.ModuleList(transforms)

        # 'p' is the probability with which the transformations will be applied.
        # It's a floating-point number between 0 and 1.
        self.p = p

    def forward(self, x):
        # Randomly decide whether to apply the transformations or not, based on probability 'p'.
        if self.p < torch.rand(1):
            # If the random number is greater than 'p', return the input as it is.
            return x

        # Assert shape is (bsz, seq_len)
        assert len(x.shape) == 2

        # Unsqueeze last dimension
        x = x.unsqueeze(-1)

        # Apply each transformation in the list to the transposed input.
        for t in self.transforms:
            x = t(x)

        # Squeeze last dimension
        x = x.squeeze(-1)

        # Finally, transpose the tensor back to its original dimension order.
        # return x_time_first.transpose(1, 2)
        return x


class Jitter(nn.Module):
    """
    The Jitter class implements a jittering transformation as described in the paper:
    'Data Augmentation for Machine Learning Algorithms' (https://arxiv.org/pdf/1706.00527.pdf).
    It adds random noise to the input data, which is a common technique for data augmentation.
    """

    def __init__(self, p, sigma=0.03):
        super().__init__()
        # 'p' is the probability with which the jitter (noise) will be applied to the input data.
        self.p = p

        # 'sigma' defines the standard deviation of the normal distribution used for generating the jitter.
        # It controls the magnitude of the noise added to the data.
        self.sigma = sigma

    def forward(self, x):
        # Randomly decide whether to apply jitter (noise) or not, based on the probability 'p'.
        if self.p < torch.rand(1):
            # If the random number is greater than 'p', return the input as it is, without adding noise.
            return x

        # Generate random noise from a normal distribution with mean 0 and standard deviation 'sigma'.
        # The size and device of the noise tensor are the same as the input tensor 'x'.
        noise = torch.normal(mean=0.0, std=self.sigma, size=x.shape, device=x.device)

        # Add the generated noise to the input data and return the result.
        return x + noise


class Scaling(nn.Module):
    """
    The Scaling class implements a scaling transformation as described in the paper:
    'Data Augmentation for Machine Learning Algorithms' (https://arxiv.org/pdf/1706.00527.pdf).
    This transformation scales the input data by a random factor, which can be useful for data augmentation.
    """

    def __init__(self, p, sigma=0.1):
        super().__init__()
        # 'p' is the probability with which the scaling will be applied to the input data.
        self.p = p

        # 'sigma' defines the standard deviation of the normal distribution used for generating the scaling factor.
        # It controls the variability of the scaling factor.
        self.sigma = sigma

    def forward(self, x):
        # Randomly decide whether to apply scaling or not, based on the probability 'p'.
        if self.p < torch.rand(1):
            # If the random number is greater than 'p', return the input as it is, without scaling.
            return x

        # Generate random scaling factors from a normal distribution with mean 1 and standard deviation 'sigma'.
        # The size of the scaling factor tensor is tailored to match the batch and time dimensions of the input tensor 'x',
        # but it has a single channel so that the same factor is applied across all channels.
        factor = torch.normal(
            mean=1.0, std=self.sigma, size=(x.shape[0], 1, x.shape[2]), device=x.device
        )

        # Multiply the input data by the scaling factors and return the result.
        return x * factor


class Rotation(nn.Module):
    """
    Rotation class is designed to randomly rotate the input data.
    It's a form of data augmentation that can be particularly useful in scenarios where
    the orientation of the data is not a defining characteristic.
    """

    def __init__(self, p):
        super().__init__()
        # 'p' is the probability of applying the rotation to the input data.
        self.p = p

    def forward(self, x):
        # Randomly decide whether to rotate the data or not, based on the probability 'p'.
        if self.p < torch.rand(1):
            # If the random number is greater than 'p', return the input as it is, without rotation.
            return x

        # Create an index for flipping, where each element has a 50% chance of being 0 or 1.
        flip_index = torch.multinomial(
            torch.tensor([0.5, 0.5], dtype=x.dtype, device=x.device),
            num_samples=x.shape[0] * x.shape[2],
            replacement=True,
        )

        # Create a tensor of ones, which will be used to flip the sign of the data based on the flip_index.
        ones = torch.ones((x.shape[0] * x.shape[2]), device=x.device)
        flip = torch.where(flip_index == 0, -ones, ones)

        # Randomly shuffle the axes along which the data will be rotated.
        rotate_axis = np.arange(x.shape[2])
        np.random.shuffle(rotate_axis)

        # Apply the flipping and rotation to the data and return the result.
        return flip.reshape(x.shape[0], 1, x.shape[2]) * x[:, :, rotate_axis]


class Permutation(nn.Module):
    """
    The Permutation class implements a data augmentation technique where the data is divided into segments,
    and these segments are then randomly permuted. This can be useful for tasks where the order of data points
    is not crucial and can help in improving the robustness of models.
    """

    def __init__(self, p, max_segments=5, seg_mode="equal"):
        super().__init__()
        # 'p' is the probability of applying the permutation to the input data.
        self.p = p

        # 'max_segments' defines the maximum number of segments into which the data can be split for permutation.
        self.max_segments = max_segments

        # 'seg_mode' determines how the segments are created: 'equal' for equal-sized segments, 'random' for random splits.
        self.seg_mode = seg_mode

    def forward(self, x):
        # Randomly decide whether to permute the data or not, based on the probability 'p'.
        if self.p < torch.rand(1):
            # If the random number is greater than 'p', return the input as it is, without permutation.
            return x

        # Create an array representing the original order of data points.
        orig_steps = np.arange(x.shape[1])

        # Randomly decide the number of segments for each batch in the data.
        num_segs = np.random.randint(1, self.max_segments, size=(x.shape[0]))

        # Initialize a tensor to hold the permuted data.
        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            if num_segs[i] > 1:
                if self.seg_mode == "random":
                    # In 'random' mode, choose random split points.
                    split_points = np.random.choice(
                        x.shape[1] - 2, num_segs[i] - 1, replace=False
                    )
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    # In 'equal' mode, split the data into roughly equal segments.
                    splits = np.array_split(orig_steps, num_segs[i])

                # Permute the segments and recombine them.
                warp = np.concatenate(np.random.permutation(splits)).ravel()
                ret[i] = pat[warp]
            else:
                # If there's only one segment, keep the data as it is.
                ret[i] = pat

        return ret


class MagnitudeWarp(nn.Module):
    """
    The MagnitudeWarp class applies a non-linear warping to the magnitude of the input data.
    This is achieved by using cubic splines to create smooth, random warp functions that are
    then applied to the input. It's a form of data augmentation useful in scenarios where the
    model needs to be robust to variations in the magnitude of the input data.
    """

    def __init__(self, p, sigma=0.2, knot=4):
        super().__init__()
        # 'p' is the probability with which the magnitude warp will be applied.
        self.p = p

        # 'sigma' controls the variability of the warp. Higher values lead to more pronounced warping.
        self.sigma = sigma

        # 'knot' is the number of points in the cubic spline used for warping.
        self.knot = knot

    def forward(self, x):
        # Decide whether to apply the warp based on the probability 'p'.
        if self.p < torch.rand(1):
            return x

        # Generate an array representing the original order of data points.
        orig_steps = np.arange(x.shape[1])

        # Generate random warps using a normal distribution centered at 1.0.
        random_warps = np.random.normal(
            loc=1.0,
            scale=self.sigma,
            size=(x.shape[0], self.knot + 2, x.shape[2]),
        )

        # Create warp steps evenly distributed across the data length.
        warp_steps = (
            np.ones((x.shape[2], 1))
            * (np.linspace(0, x.shape[1] - 1.0, num=self.knot + 2))
        ).T

        # Initialize a tensor to hold the warped data.
        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            # For each dimension, create a cubic spline based on the warp steps and random warps,
            # and apply it to the original steps to get the warper.
            warper = np.array(
                [
                    CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
                    for dim in range(x.shape[2])
                ]
            ).T

            # Apply the warper to the pattern and store it in the result tensor.
            ret[i] = pat * torch.from_numpy(warper).float().to(x.device)

        return ret


class TimeWarp(nn.Module):
    """
    The TimeWrap class applies a non-linear warping to the time axis of the input data.
    This is achieved by using cubic splines to create smooth, random warp functions that
    distort the time dimension of the input. It's a form of data augmentation useful for
    tasks where the model needs to be robust to variations in the timing of the input data.
    """

    def __init__(self, p, sigma=0.2, knot=4):
        super().__init__()
        # 'p' is the probability with which the time warp will be applied.
        self.p = p

        # 'sigma' controls the variability of the warp. Higher values lead to more pronounced warping.
        self.sigma = sigma

        # 'knot' is the number of points in the cubic spline used for warping.
        self.knot = knot

    def forward(self, x):
        # Decide whether to apply the warp based on the probability 'p'.
        if self.p < torch.rand(1):
            return x

        # Generate an array representing the original time steps of the data.
        orig_steps = np.arange(x.shape[1])

        # Generate random warps using a normal distribution centered at 1.0.
        random_warps = np.random.normal(
            loc=1.0,
            scale=self.sigma,
            size=(x.shape[0], self.knot + 2, x.shape[2]),
        )

        # Create warp steps evenly distributed across the data length.
        warp_steps = (
            np.ones((x.shape[2], 1))
            * (np.linspace(0, x.shape[1] - 1.0, num=self.knot + 2))
        ).T

        # Initialize a tensor to hold the time-warped data.
        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                # Create a cubic spline based on the warp steps and random warps to generate the time warp.
                time_warp = CubicSpline(
                    warp_steps[:, dim],
                    warp_steps[:, dim] * random_warps[i, :, dim],
                )(orig_steps)
                # Scale the time warp to fit the original data length.
                scale = (x.shape[1] - 1) / time_warp[-1]
                wrap = np.interp(
                    orig_steps,
                    np.clip(scale * time_warp, 0, x.shape[1] - 1),
                    pat[:, dim].cpu().numpy(),
                ).T
                # Apply the time warp to the corresponding dimension of the data.
                ret[i, :, dim] = torch.from_numpy(wrap).float().to(x.device)

        return ret


class WindowSlice(nn.Module):
    """
    The WindowSlice class implements a data augmentation technique where a slice of the input data
    is stretched to fill the entire length of the input. This technique is useful for training models
    to focus on local features of the data and can be found in literature such as:
    'Time Series Data Augmentation for Deep Learning: A Survey' (https://halshs.archives-ouvertes.fr/halshs-01357973/document).
    """

    def __init__(self, p, reduce_ratio=0.9):
        super().__init__()
        # 'p' is the probability of applying the window slicing to the input data.
        self.p = p

        # 'reduce_ratio' determines the size of the slice relative to the original data.
        self.reduce_ratio = reduce_ratio

    def forward(self, x):
        # Decide whether to apply the slice based on the probability 'p'.
        if self.p < torch.rand(1):
            return x

        # Calculate the target length of the slice.
        target_len = np.ceil(self.reduce_ratio * x.shape[1]).astype(int)
        if target_len >= x.shape[1]:
            return x

        # Randomly select start points for the slice in each batch.
        starts = np.random.randint(
            low=0, high=x.shape[1] - target_len, size=(x.shape[0])
        ).astype(int)
        ends = (target_len + starts).astype(int)

        # Initialize a tensor to hold the sliced and stretched data.
        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                # Interpolate the slice to stretch it across the original length of the data.
                warp = np.interp(
                    np.linspace(0, target_len, num=x.shape[1]),
                    np.arange(target_len),
                    pat[starts[i] : ends[i], dim].cpu().numpy(),
                ).T
                # Apply the stretched slice to the corresponding dimension of the data.
                ret[i, :, dim] = torch.from_numpy(warp).float().to(x.device)
        return ret


class WindowWarp(nn.Module):
    """
    The WindowWarp class implements a data augmentation technique where a segment (window) of the input data
    is selected and warped in size. This technique is useful for simulating variations in the speed or rate
    of the data within a certain window, as discussed in:
    'Time Series Data Augmentation for Deep Learning: A Survey' (https://halshs.archives-ouvertes.fr/halshs-01357973/document).
    """

    def __init__(self, p, window_ratio=0.1, scales=[0.5, 2.0]):
        super().__init__()
        # 'p' is the probability of applying the window warp to the input data.
        self.p = p

        # 'window_ratio' determines the size of the window relative to the original data.
        self.window_ratio = window_ratio

        # 'scales' are the possible scaling factors to be applied to the window.
        self.scales = scales

    def forward(self, x):
        # Decide whether to apply the warp based on the probability 'p'.
        if self.p < torch.rand(1):
            return x

        # Randomly choose a scaling factor for each batch in the data.
        warp_scales = np.random.choice(self.scales, x.shape[0])

        # Calculate the size of the warp window.
        warp_size = np.ceil(self.window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)

        # Randomly select start points for the window in each batch.
        window_starts = np.random.randint(
            low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])
        ).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        # Initialize a tensor to hold the window-warped data.
        ret = torch.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                # Isolate the segments before, within, and after the window.
                start_seg = pat[: window_starts[i], dim].cpu().numpy()
                window_seg = np.interp(
                    np.linspace(
                        0,
                        warp_size - 1,
                        num=int(warp_size * warp_scales[i]),
                    ),
                    window_steps,
                    pat[window_starts[i] : window_ends[i], dim].cpu().numpy(),
                )
                end_seg = pat[window_ends[i] :, dim].cpu().numpy()

                # Concatenate the segments and stretch them to fit the original data length.
                warped = np.concatenate((start_seg, window_seg, end_seg))
                warp = np.interp(
                    np.arange(x.shape[1]),
                    np.linspace(0, x.shape[1] - 1.0, num=warped.size),
                    warped,
                ).T

                # Apply the window warp to the corresponding dimension of the data.
                ret[i, :, dim] = torch.from_numpy(warp).float().to(x.device)
        return ret
