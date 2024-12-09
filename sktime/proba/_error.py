"""Error message for attempted import of sktime proba module."""


def _proba_error(*args, **kwargs):
    """Raise error for attempted import of sktime proba module."""
    raise ImportError(
        "The sktime proba module has been deprecated and will be removed in 0.38.0. "
        "Please import from skpro.distributions instead of sktime.proba. "
        "Imports from sktime.proba will be possible until 0.38.0 if skpro "
        "is installed, and will raise an error otherwise. "
        "After 0.38.0, imports from sktime.proba will raise an error regardless."
    )
