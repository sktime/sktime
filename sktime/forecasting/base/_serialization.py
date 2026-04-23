# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mixin for serialization of zero-shot models."""

class _ZeroShotSerializationMixin:
    """Mixin for zero-shot models that use multiton caching.

    Ensures that unpicklable model pipelines and cached multiton instances
    are excluded from the pickle state and reloaded on demand.
    """

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable components."""
        state = self.__dict__.copy()

        # List of attributes to exclude from serialization
        # These are usually the heavy/unpicklable model objects
        to_exclude = [
            "model_pipeline",
            "_model_pipeline",
            "predictor_",
            "estimator_",
            "tfm",  # for TimesFM
            "model",  # generic
        ]

        for attr in to_exclude:
            if attr in state:
                state[attr] = None

        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)
