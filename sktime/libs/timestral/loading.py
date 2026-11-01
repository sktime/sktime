"""Load exported checkpoints: a directory with `config.json` and `model.safetensors`, either on
disk or in a Hugging Face Hub repo."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HFValidationError, RepositoryNotFoundError
from safetensors.torch import load_file

from sktime.libs.timestral.config import TimestralConfig
from sktime.libs.timestral.model import TimestralModel

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def resolve_checkpoint(
    path_or_repo_id: str | Path,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local directory holding `config.json` and `model.safetensors`.

    An existing directory is used as is. Anything else is a Hugging Face Hub repo id
    (e.g. `org/timestral-108m`): its two files are downloaded to the Hub cache, or read from it
    when `local_files_only` (or `HF_HUB_OFFLINE=1`) is set.
    """
    path = Path(path_or_repo_id).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        raise NotADirectoryError(
            f"{str(path)!r} is a file; pass the export directory holding {CONFIG_FILE} and {WEIGHTS_FILE}"
        )
    try:
        return Path(
            snapshot_download(
                str(path_or_repo_id),
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=[CONFIG_FILE, WEIGHTS_FILE],
            )
        )
    except GatedRepoError:
        raise  # the Hub's message says how to request access
    except (RepositoryNotFoundError, HFValidationError) as e:
        raise FileNotFoundError(
            f"{str(path_or_repo_id)!r} is neither a local directory nor a Hugging Face Hub repo you can access"
        ) from e


def load_model(
    path_or_repo_id: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    **hub_kwargs,
) -> TimestralModel:
    """Build the model from `config.json` and load `model.safetensors`.

    `path_or_repo_id` is a local directory or a Hugging Face Hub repo id; `hub_kwargs` go to
    `resolve_checkpoint`. Weights load with `strict=True`. Inference was validated in float32.
    """
    path = resolve_checkpoint(path_or_repo_id, **hub_kwargs)
    config = TimestralConfig.load(path / CONFIG_FILE)
    model = TimestralModel(config)
    model.load_state_dict(load_file(path / WEIGHTS_FILE, device="cpu"), strict=True)
    model.to(device=device, dtype=dtype)
    model.transformer.rope.rebuild_inv_freq()
    return model.eval()
