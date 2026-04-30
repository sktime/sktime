"""Serialization helpers for forecasting estimators."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)


class _CachedModelSerializationMixin:
    """Exclude cached model handles from pickle state and lazily restore them."""

    _cached_model_fields = ()

    def __getstate__(self):
        """Return state for pickling without non-serializable cached models."""
        state = self.__dict__.copy()
        for field in self._cached_model_fields:
            if field in state:
                state[field] = None
        return state

    def __setstate__(self, state):
        """Restore state from an unpickled state dictionary."""
        self.__dict__.update(state)

    def _get_cached_model_loaders(self):
        """Return mapping of cached model attributes to loader callables."""
        return {}

    def _ensure_cached_models_loaded(self):
        """Restore cached model handles after unpickling if estimator is fitted."""
        if not getattr(self, "_is_fitted", False):
            return

        for field, loader in self._get_cached_model_loaders().items():
            if getattr(self, field, None) is None:
                setattr(self, field, loader())
