"""Utilities for safely cloning and serializing torch.nn.Module state_dicts.

This module provides small helpers to clone a PyTorch module's state_dict
(to CPU, detached, and cloned tensors), serialize it to bytes, and restore it.
Imports are lazy/safe: functions raise ImportError only when torch is actually
required, so the module can be imported in environments without torch.

Functions
---------
clone_state_dict(module)
    Return a CPU-detached clone of module.state_dict().
load_state_dict_into(module, state_dict)
    Load a state_dict (CPU tensors allowed) into a module.
module_to_bytes(module)
    Serialize a module's cloned state_dict to bytes.
bytes_to_state_dict(b)
    Deserialize bytes produced by module_to_bytes back to a state_dict.
is_torch_module(obj)
    Return True if obj is an instance of torch.nn.Module (safe if torch missing).

Notes
-----
These helpers are intended to avoid deepcopy/pickling of nn.Module objects by
transferring only the state_dict. Use them to make model persistence and cloning
robust across PyTorch versions.
"""

import io

import torch
from skbase.utils.dependencies import _safe_import
from torch import nn


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU-cloned copy of module.state_dict()."""
    sd = module.state_dict()
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def load_state_dict_into(
    module: nn.Module, state_dict: dict[str, torch.Tensor], map_location=None
):
    """Load a state_dict into module (thin wrapper around module.load_state_dict)."""
    module.load_state_dict(state_dict)


def module_to_bytes(module: nn.Module) -> bytes:
    """Serialize a module's cloned state_dict to bytes (torch.save of cloned state)."""
    buf = io.BytesIO()
    torch.save(clone_state_dict(module), buf)
    return buf.getvalue()


def bytes_to_state_dict(b: bytes):
    """Deserialize bytes produced by module_to_bytes into a CPU state_dict."""
    buf = io.BytesIO(b)
    return torch.load(buf, map_location="cpu")


def is_torch_module(obj) -> bool:
    """Return True if obj is an instance of torch.nn.Module (safe if torch missing)."""
    torch = _safe_import("torch")
    if torch is None:
        return False
    return isinstance(obj, torch.nn.Module)
