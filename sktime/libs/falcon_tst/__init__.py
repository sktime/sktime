"""Python implementation of Falcon-TST.

Unofficial fork of the ``ant-intl/Falcon-TST_Large`` model code from
Hugging Face, available at https://huggingface.co/ant-intl/Falcon-TST_Large.

sktime migration: 2026, Jun
"""

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", "transformers", "einops", severity="none"):
    from sktime.libs.falcon_tst.configuration_FalconTST import FalconTSTConfig
    from sktime.libs.falcon_tst.modeling_FalconTST import FalconTSTForPrediction
else:

    class FalconTSTConfig:
        """Placeholder for FalconTSTConfig."""

    class FalconTSTForPrediction:
        """Placeholder for FalconTSTForPrediction."""
