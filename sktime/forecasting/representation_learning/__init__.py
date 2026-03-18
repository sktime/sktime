"""Shared components for representation-learning-based forecasters.

Representation learning here means forecasters whose core mechanism is a
latent representation of the series (factors, embeddings, or other codes),
from which forecasts are derived. That includes:

- Conventional factor models (e.g. DFM) and their state-space form.
- Autoencoder or VAE-based models (e.g. DeepDynamicFactor).
- Transformer or foundation-model-based forecasters that use learned embeddings.

This subpackage holds reusable building blocks (datasets, state-space helpers,
autoencoder training utilities, etc.) shared by such forecasters. The forecaster
classes themselves live in their own modules under forecasting/.
"""

__all__: list[str] = []
