# ruff: noqa
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


class RevIN(nn.Module):
    def __init__(self, dim=-1, std_min=1e-5, max_val=100, use_sinh=False):
        super().__init__()
        self.dim = dim
        self.std_min = std_min
        self.max_val = max_val
        self.use_sinh = use_sinh

    def fit_transform(self, x, mask=None):
        with torch.autocast(device_type="cuda", enabled=False):
            self._get_statistics(x, mask)
            return self.transform(x)

    def transform(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            x = (x - self.mean) / self.std
            if self.use_sinh:
                x = torch.asinh(x)
            return x

    def inverse_transform(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            if self.use_sinh:
                x = torch.sinh(x)
            if x.ndim != self.mean.ndim:
                x = x * self.std.unsqueeze(1) + self.mean.unsqueeze(1)
            else:
                x = x * self.std + self.mean

            return x

    def get_statistics(self):
        return self.mean, self.std

    def _get_statistics(self, x, mask=None):
        if mask is None:
            self.mean = x.mean(dim=self.dim, keepdim=True)
            std = x.std(dim=self.dim, keepdim=True)
            self.std = torch.where(std > self.std_min, std, torch.ones_like(std))
        else:
            mask = mask.bool()
            unmask = (~mask).float()
            count = unmask.sum(dim=self.dim, keepdim=True).clamp(
                min=1
            )  # avoid division by zero
            x_mean = (x * unmask).sum(dim=self.dim, keepdim=True) / count
            x_std = (((x - x_mean) * unmask) ** 2).sum(
                dim=self.dim, keepdim=True
            ) / count
            x_std = x_std.sqrt()
            x_std = torch.where(x_std > self.std_min, x_std, torch.ones_like(x_std))
            self.mean = x_mean
            self.std = x_std


class CausalRevIN(nn.Module):
    def __init__(self, dim=-1, std_min=1e-5, max_val=100):
        """
        Causal RevIN implementation to enable parallel predictions during training of FlowState

        :param eps: a value added for numerical stability
        :param with_missing (bool): whether contiguous patch masking (CPM) is used or not, interpreting nans as missing values
        """
        super().__init__()
        self.dim = dim
        self.std_min = std_min
        self.max_val = max_val

    def fit_transform(self, x, mask=None):
        self._get_statistics(x, mask)
        return self.transform(x)

    def transform(self, x):
        return torch.clamp(
            (x - self.mean) / self.std, min=-self.max_val, max=self.max_val
        )

    def inverse_transform(self, x):
        if x.ndim == 2:
            return x * self.std + self.mean
        elif x.ndim == 3:
            return x * self.std.unsqueeze(-1) + self.mean.unsqueeze(-1)
        else:
            raise ValueError(f"Invalid input dimension: {x.shape}")

    def get_statistics(self):
        return self.mean, self.std

    def _get_statistics(self, x, mask=None):
        if mask is not None:
            n = torch.cumsum(1 - mask.float(), dim=1)
            n = torch.where(n == 0, 1.0, n)
        else:
            n = torch.arange(1, x.shape[1] + 1, device=x.device)
        self.mean = (torch.cumsum(x, dim=1) / n).detach()
        mask = 1 - mask.float() if mask is not None else 1
        self.std = torch.sqrt(
            torch.cumsum(((x - self.mean) * mask) ** 2, 1) / n
        ).detach()
        self.std = torch.where(
            self.std > self.std_min, self.std, torch.ones_like(self.std)
        )

    def set_statistics(self, mean, std):
        self.mean = mean
        self.std = std
