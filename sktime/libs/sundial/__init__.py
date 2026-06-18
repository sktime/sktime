"""Python implementation of Sundial.

Unofficial fork of the ``thuml/sundial-base-128m`` model code from Hugging Face,
available at https://huggingface.co/thuml/sundial-base-128m.

sktime migration: 2026, Jun
"""

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", "transformers", severity="none"):
    from sktime.libs.sundial.configuration_sundial import SundialConfig
    from sktime.libs.sundial.modeling_sundial import SundialForPrediction
else:

    class SundialConfig:
        """Placeholder for SundialConfig."""

    class SundialForPrediction:
        """Placeholder for SundialForPrediction."""
