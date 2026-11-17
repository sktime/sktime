# ruff: noqa
"""Trimmed from models/tools.py: only what modeling_patchtst_fm.py imports."""


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    no_grad_params = total_params - grad_params
    return total_params, grad_params, no_grad_params
