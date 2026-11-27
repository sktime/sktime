# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""State-aware clone and set_params for arbitrary objects."""

__author__ = ["SimonBlanke"]

from sktime.base._base import _safe_clone

_VALID_TARGET_STATES = ("new", "pretrained")


def _check_target_state(state):
    """Raise ValueError if state is not a valid target state."""
    if state not in _VALID_TARGET_STATES:
        raise ValueError(
            f"target state must be one of {_VALID_TARGET_STATES}, but found {state!r}"
        )


def _clone_at(obj, state):
    """Clone any object, retaining state up to ``state`` where supported.

    For use by composites on their components. Calls ``obj._clone_at(state)``
    if ``obj`` implements it, otherwise falls back to ``_safe_clone(obj)``,
    i.e., ``obj.clone()``, or sklearn ``clone`` for non-``sktime`` objects.

    Parameters
    ----------
    obj : object to clone, e.g., an estimator held by a composite
    state : str, "pretrained" or "new"
        Upper bound on the retained state tier, see ``_PretrainMixin._clone_at``.

    Returns
    -------
    clone : new object of same type, without shared references to ``obj``.

    Raises
    ------
    ValueError
        If ``state`` is not one of ``"pretrained"``, ``"new"``.
    """
    _check_target_state(state)
    if hasattr(obj, "_clone_at"):
        return obj._clone_at(state)
    return _safe_clone(obj)


def _set_params_at(obj, state, params):
    """Set parameters of any object, retaining state up to ``state`` if supported.

    For use by composites on their components. Calls
    ``obj._set_params_at(state, params)`` if ``obj`` implements it, otherwise
    falls back to ``obj.set_params(**params)``.

    Parameters
    ----------
    obj : object to set parameters of, e.g., an estimator held by a composite
    state : str, "pretrained" or "new"
        Upper bound on the retained state tier, see ``_PretrainMixin._reset_at``.
    params : dict
        Parameters as in ``set_params``, passed as an explicit dict.

    Returns
    -------
    obj : reference to ``obj``, with parameters set.

    Raises
    ------
    ValueError
        If ``state`` is not one of ``"pretrained"``, ``"new"``.
    """
    _check_target_state(state)
    if hasattr(obj, "_set_params_at"):
        return obj._set_params_at(state, params)
    return obj.set_params(**params)
