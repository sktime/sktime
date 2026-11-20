"""Vendored Tabby model implementation.

Weights and vendored code are CC BY-NC 4.0 (non-commercial use only), copied
from https://huggingface.co/paris-noah/Tabby at revision
``1b2381d7f1b0982aa1d5f09ce01b99eb4ee57e01``. Weights are not bundled with
``sktime``.
"""

from .configuration_tabby import TabbyConfig
from .modeling_tabby import TabbyForForecasting

__all__ = ["TabbyConfig", "TabbyForForecasting"]
