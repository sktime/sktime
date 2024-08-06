"""Masking module for momentfm."""

from typing import Optional

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies(["torch"], severity="none"):
    import torch

    class Masking:
        """Masking Module."""

        def __init__(
            self,
            mask_ratio: float = 0.3,
            patch_len: int = 8,
            stride: Optional[int] = None,
        ):
            """Masking class for momentfm.

            Indices with 0 mask are hidden, and with 1 are observed.
            """
            self.mask_ratio = mask_ratio
            self.patch_len = patch_len
            self.stride = patch_len if stride is None else stride

        @staticmethod
        def convert_seq_to_patch_view(
            mask: torch.Tensor, patch_len: int = 8, stride: Optional[int] = None
        ):
            """Convert sequence to patch function.

            Input:
                mask : torch.Tensor of shape [batch_size x seq_len]
            Output
                mask : torch.Tensor of shape [batch_size x n_patches]
            """
            stride = patch_len if stride is None else stride
            mask = mask.unfold(dimension=-1, size=patch_len, step=stride)
            # mask : [batch_size x n_patches x patch_len]
            return (mask.sum(dim=-1) == patch_len).long()

        @staticmethod
        def convert_patch_to_seq_view(
            mask: torch.Tensor,
            patch_len: int = 8,
        ):
            """Convert patch to sequence function.

            Input:
                mask : torch.Tensor of shape [batch_size x n_patches]
            Output:
                mask : torch.Tensor of shape [batch_size x seq_len]
            """
            return mask.repeat_interleave(patch_len, dim=-1)

        def generate_mask(
            self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None
        ):
            """Generate Mask Function.

            Input:
                x : torch.Tensor of shape
                [batch_size x n_channels x n_patches x patch_len] or
                [batch_size x n_channels x seq_len]
                input_mask: torch.Tensor of shape [batch_size x seq_len] or
                [batch_size x n_patches]
            Output:
                mask : torch.Tensor of shape [batch_size x seq_len]
            """
            if x.ndim == 4:
                return self._mask_patch_view(x, input_mask=input_mask)
            elif x.ndim == 3:
                return self._mask_seq_view(x, input_mask=input_mask)

        def _mask_patch_view(self, x, input_mask=None):
            """Return mask for inputted patch view.

            Input:
                x : torch.Tensor of shape
                [batch_size x n_channels x n_patches x patch_len]
                input_mask: torch.Tensor of shape [batch_size x seq_len]
            Output:
                mask : torch.Tensor of shape [batch_size x n_patches]
            """
            input_mask = self.convert_seq_to_patch_view(
                input_mask, self.patch_len, self.stride
            )
            n_observed_patches = input_mask.sum(dim=-1, keepdim=True)  # batch_size x 1

            batch_size, _, n_patches, _ = x.shape
            len_keep = torch.ceil(n_observed_patches * (1 - self.mask_ratio)).long()
            noise = torch.rand(
                batch_size, n_patches, device=x.device
            )  # noise in [0, 1], batch_size x n_channels x n_patches
            noise = torch.where(
                input_mask == 1, noise, torch.ones_like(noise)
            )  # only keep the noise of observed patches

            # Sort noise for each sample
            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # Ascend: small is keep, large is remove
            ids_restore = torch.argsort(
                ids_shuffle, dim=1
            )  # ids_restore: [batch_size x n_patches]

            # Generate the binary mask: 0 is keep, 1 is remove
            mask = torch.zeros(
                [batch_size, n_patches], device=x.device
            )  # mask: [batch_size x n_patches]
            for i in range(batch_size):
                mask[i, : len_keep[i]] = 1

            # Unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return mask.long()

        def _mask_seq_view(self, x, input_mask=None):
            """Return mask for inputted tensor.

            Input:
                x : torch.Tensor of shape
                [batch_size x n_channels x seq_len]
                input_mask: torch.Tensor of shape [batch_size x seq_len]
            Output:
                mask : torch.Tensor of shape [batch_size x seq_len]
            """
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            mask = self._mask_patch_view(x, input_mask=input_mask)
            return self.convert_patch_to_seq_view(mask, self.patch_len).long()

else:

    class Masking:
        """Dummy class if torch is unavailable."""

        pass
