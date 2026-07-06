import json
from pathlib import Path

from sktime.utils.dependencies import _safe_import  # [sktime] soft-dep isolation

from ..model.dynamix import DynaMix

torch = _safe_import("torch")
save_file = _safe_import("safetensors.torch.save_file")
load_file = _safe_import("safetensors.torch.load_file")
hf_hub_download = _safe_import("huggingface_hub.hf_hub_download")


def create_checkpoint_directories(save_path, args=None):
    """
    Create directories for saving checkpoints, plots, and configurations.

    Args:
        save_path: Base path for saving
        args: Optional arguments to save as configuration

    Returns
    -------
        save_dir: Path object for the base save directory
        checkpoint_dir: Path object for the checkpoint directory
        plots_dir: Path object for the plots directory
    """
    save_dir = Path(save_path)
    checkpoint_dir = save_dir / "checkpoints"
    plots_dir = save_dir / "plots"

    # Create directories if they don't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration if provided
    if args is not None:
        config = {k: v for k, v in vars(args).items()}
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    return save_dir, checkpoint_dir, plots_dir


def save_model_pt(model_path, model):
    """
    Save only a model's state dict in PyTorch format.

    Args:
        model_path: Path where to save the model
        model: Model to save
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save only the model state dict
    torch.save(model.state_dict(), model_path)


def load_model_pt(model_path, model, device="cpu"):
    """
    Load a model state dict into a model from PyTorch format.

    Args:
        model_path: Path to the model file
        model: Model to load state_dict into
        device: Device to load the model to (default: 'cpu')

    Returns
    -------
        model: Loaded model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    return model


def save_model(model_path, model):
    """
    Save only a model's state dict in SafeTensors format.

    Args:
        model_path: Path where to save the model
        model: Model to save
    """
    model_path = Path(model_path)
    # Change extension to .safetensors
    if model_path.suffix != ".safetensors":
        model_path = model_path.with_suffix(".safetensors")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Get state dict and ensure all tensors are contiguous
    state_dict = model.state_dict()
    for key in state_dict:
        if (
            isinstance(state_dict[key], torch.Tensor)
            and not state_dict[key].is_contiguous()
        ):
            state_dict[key] = state_dict[key].contiguous()

    # Save using safetensors
    save_file(state_dict, model_path)


def load_model(model_path, model, device="cpu"):
    """
    Load a model state dict into a model from SafeTensors format.

    Args:
        model_path: Path to the model file
        model: Model to load state_dict into
        device: Device to load the model to (default: 'cpu')

    Returns
    -------
        model: Loaded model
    """
    model_path = Path(model_path)
    # Change extension to .safetensors if not already
    if model_path.suffix != ".safetensors":
        safetensors_path = model_path.with_suffix(".safetensors")
        # Check if safetensors version exists, otherwise try original path
        if safetensors_path.exists():
            model_path = safetensors_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the state dict using safetensors if it has .safetensors extension
    if model_path.suffix == ".safetensors":
        state_dict = load_file(model_path, device=device)
    else:
        # Fallback to PyTorch loading for backward compatibility
        state_dict = torch.load(model_path, map_location=device)

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)

    return model


"""
Huggingface model loader
"""


def load_hf_model_config(model_name):
    """Load model configuration from HuggingFace Hub"""
    config_path = hf_hub_download(
        repo_id="DurstewitzLab/dynamix",
        filename="config_" + model_name.replace("dynamix-", "") + ".json",
    )

    with open(config_path) as f:
        model_config = json.load(f)

    return model_config


def load_hf_model(model_name):
    """Load a specific DynaMix model with its configuration"""
    try:
        # Load model configuration
        model_config = load_hf_model_config(model_name)
        architecture = model_config["architecture"]

        # Extract hyperparameters from config
        M = architecture["M"]
        N = architecture["N"]
        EXPERTS = architecture["Experts"]
        P = architecture["P"]
        HIDDEN_DIM = architecture["hidden_dim"]
        expert_type = architecture["expert_type"]
        probabilistic_expert = architecture["probabilistic_expert"]

        # Create model with config parameters
        model = DynaMix(
            M=M,
            N=N,
            Experts=EXPERTS,
            expert_type=expert_type,
            P=P,
            hidden_dim=HIDDEN_DIM,
            probabilistic_expert=probabilistic_expert,
        )

        # Load model weights
        model_path = hf_hub_download(
            repo_id="DurstewitzLab/dynamix", filename=model_name + ".safetensors"
        )
        model_state_dict = load_file(model_path)
        model.load_state_dict(model_state_dict)
        model.eval()

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise ValueError(f"Model {model_name} not found")

    return model
