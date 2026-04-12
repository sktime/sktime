"""Mixin for serialization of zero-shot foundation model forecasters.

Provides ``__getstate__`` and ``__setstate__`` for forecasters that use
a cached singleton (multiton) pattern to hold a loaded model pipeline.

This mixin is intended for zero-shot foundation model forecasters such as
``ChronosForecaster`` and ``Chronos2Forecaster``, where the model pipeline
is an unpickleable object (e.g., a PyTorch model) held in a module-level
singleton registry.

The mixin handles serialization by:
- Excluding ``model_pipeline`` from the pickle state
- Restoring ``model_pipeline`` lazily on first use after unpickling
  via ``_ensure_model_pipeline_loaded``, which must be implemented
  by the subclass.
"""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["NestroyMusoke"]

__all__ = ["_ZeroShotSerializationMixin"]


class _ZeroShotSerializationMixin:
    """Mixin for serialization of zero-shot foundation model forecasters.

    Handles pickling and unpickling for forecasters that store a loaded
    model pipeline (e.g., a PyTorch model) in a cached singleton.

    The model pipeline is excluded from the pickle state and restored
    lazily on first use after unpickling.

    Subclasses must implement ``_load_pipeline`` and
    ``_ensure_model_pipeline_loaded``.

    Examples
    --------
    Typical usage in a zero-shot foundation model forecaster:

    >>> from sktime.forecasting._foundation_model_mixin import (
    ...     _ZeroShotSerializationMixin,
    ... )
    >>> from sktime.forecasting.base import BaseForecaster
    >>>
    >>> class MyFoundationForecaster(
    ...     _ZeroShotSerializationMixin, BaseForecaster
    ... ):
    ...     def _load_pipeline(self):
    ...         # load and return the model pipeline
    ...         ...
    ...
    ...     def _ensure_model_pipeline_loaded(self):
    ...         if (
    ...             not hasattr(self, "model_pipeline")
    ...             or self.model_pipeline is None
    ...         ):
    ...             if hasattr(self, "_is_fitted") and self._is_fitted:
    ...                 self.model_pipeline = self._load_pipeline()
    """

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable model pipeline.

        The ``model_pipeline`` attribute holds a loaded PyTorch model which
        cannot be pickled directly. It is excluded from the state and will
        be restored lazily on first use after unpickling via
        ``_ensure_model_pipeline_loaded``.

        Returns
        -------
        state : dict
            Copy of ``__dict__`` with ``model_pipeline`` set to ``None``.
        """
        state = self.__dict__.copy()
        state["model_pipeline"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary.

        Restores all attributes from ``state``. The ``model_pipeline``
        will be ``None`` after unpickling and is restored lazily on first
        use via ``_ensure_model_pipeline_loaded``.

        Parameters
        ----------
        state : dict
            State dictionary from ``__getstate__``.
        """
        self.__dict__.update(state)
