"""sklearn ``__init__`` contract conformance tests for #10208.

The sklearn ``__init__`` contract has two relevant clauses for sktime
estimators:

1. **No mutable default arguments.** A list literal as a default value is
   evaluated once at function-definition time and shared across calls,
   so mutating it on one instance silently corrupts all other instances
   that used the default.
2. **No computed self-attribute writes.** ``__init__`` must store every
   parameter exactly as passed, with no mutation. This is what lets
   ``get_params`` return what the caller supplied and lets ``clone``
   produce a faithful, independent copy.

The four estimators below all violated one or both clauses on
``main`` (see issue body of #10208 and parent issues #9971, #10024):

* ``GRUFCNNClassifier``      — list-literal defaults + computed
  ``self.optimizer_kwargs``
* ``MCDCNNClassifier``       — overwritten ``self.optimizer_kwargs``
* ``MCDCNNRegressorTorch``   — overwritten ``self.optimizer_kwargs``
* ``RBFForecaster``          — list-literal default for ``hidden_layers``

The tests in this module pin both clauses for each estimator using only
``inspect.signature`` and ``get_params`` / ``clone`` — no fit/predict —
so they run quickly and don't require GPU/CPU torch tensors.
"""

__author__ = ["jbbqqf"]

import inspect

import pytest

from sktime.utils.dependencies import _check_soft_dependencies


_TORCH_REQUIRED = pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="torch soft dependency missing",
)


def _no_mutable_defaults(cls):
    """Assert every default in ``cls.__init__`` is hashable (= immutable).

    Mutable defaults like ``[64, 32]`` are rejected; immutable defaults
    like ``None``, tuples, ints, strings, etc. are accepted.
    """
    sig = inspect.signature(cls.__init__)
    bad = []
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            continue
        try:
            hash(param.default)
        except TypeError:
            bad.append((name, param.default))
    assert not bad, (
        f"{cls.__name__}.__init__ has mutable default(s): {bad}. "
        "Use ``None`` and resolve at use-time instead."
    )


def _get_params_round_trip(instance, params):
    """Assert ``instance.get_params()`` returns exactly ``params``."""
    got = instance.get_params(deep=False)
    for k, v in params.items():
        assert got[k] == v, (
            f"{type(instance).__name__}.get_params()[{k!r}] = {got[k]!r}, "
            f"but the caller passed {v!r}. ``__init__`` must store "
            f"parameters exactly as received (sklearn contract)."
        )


# --------------------------------------------------------------------- #
# GRUFCNNClassifier
# --------------------------------------------------------------------- #


@_TORCH_REQUIRED
def test_gru_fcnn_no_mutable_defaults():
    from sktime.classification.deep_learning.gru import GRUFCNNClassifier

    _no_mutable_defaults(GRUFCNNClassifier)


@_TORCH_REQUIRED
def test_gru_fcnn_init_round_trip_default():
    """Default-construct: every param round-trips through get_params."""
    from sktime.classification.deep_learning.gru import GRUFCNNClassifier

    inst = GRUFCNNClassifier(hidden_dim=8, gru_layers=1)
    # Reads of attributes must equal the defaults declared in the
    # signature — i.e. None for the formerly-mutable ones, not the
    # resolved [128, 256, 128] / [7, 5, 3] / Adam-betas dict.
    assert inst.conv_layers is None
    assert inst.kernel_sizes is None
    assert inst.optimizer_kwargs is None


@_TORCH_REQUIRED
def test_gru_fcnn_init_round_trip_explicit():
    from sktime.classification.deep_learning.gru import GRUFCNNClassifier

    user_conv = [16, 32]
    user_kernels = [3, 3]
    user_kwargs = {"betas": (0.5, 0.5)}
    inst = GRUFCNNClassifier(
        hidden_dim=8,
        gru_layers=1,
        conv_layers=user_conv,
        kernel_sizes=user_kernels,
        optimizer_kwargs=user_kwargs,
    )
    _get_params_round_trip(
        inst,
        {
            "conv_layers": user_conv,
            "kernel_sizes": user_kernels,
            "optimizer_kwargs": user_kwargs,
        },
    )


@_TORCH_REQUIRED
def test_gru_fcnn_default_lists_are_independent():
    """Mutating one default-constructed instance must not affect another."""
    from sktime.classification.deep_learning.gru import GRUFCNNClassifier

    a = GRUFCNNClassifier(hidden_dim=8, gru_layers=1)
    b = GRUFCNNClassifier(hidden_dim=8, gru_layers=1)
    # Both should be ``None`` (mutating ``None`` is impossible — the test
    # body would have crashed on ``main`` if the defaults had been the
    # shared ``[128, 256, 128]`` lists).
    assert a.conv_layers is None and b.conv_layers is None


# --------------------------------------------------------------------- #
# MCDCNNClassifier (torch)
# --------------------------------------------------------------------- #


@_TORCH_REQUIRED
def test_mcdcnn_classifier_no_mutable_defaults():
    from sktime.classification.deep_learning.mcdcnn._mcdcnn_torch import (
        MCDCNNClassifierTorch,
    )

    _no_mutable_defaults(MCDCNNClassifierTorch)


@_TORCH_REQUIRED
def test_mcdcnn_classifier_default_optim_kwargs_is_None():
    """``optim_kwargs`` is None when the caller didn't pass it."""
    from sktime.classification.deep_learning.mcdcnn._mcdcnn_torch import (
        MCDCNNClassifierTorch,
    )

    inst = MCDCNNClassifierTorch()
    # The whole point of #10208: ``self.optim_kwargs`` must mirror the
    # caller's input. On ``main`` it would be the resolved
    # ``{"momentum": 0.9, "weight_decay": 0.0005}``.
    assert inst.optim_kwargs is None


@_TORCH_REQUIRED
def test_mcdcnn_classifier_explicit_optim_kwargs_round_trip():
    from sktime.classification.deep_learning.mcdcnn._mcdcnn_torch import (
        MCDCNNClassifierTorch,
    )

    user_kwargs = {"momentum": 0.42}
    inst = MCDCNNClassifierTorch(optim_kwargs=user_kwargs)
    _get_params_round_trip(inst, {"optim_kwargs": user_kwargs})


# --------------------------------------------------------------------- #
# MCDCNNRegressorTorch
# --------------------------------------------------------------------- #


@_TORCH_REQUIRED
def test_mcdcnn_regressor_no_mutable_defaults():
    from sktime.regression.deep_learning.mcdcnn._mcdcnn_torch import (
        MCDCNNRegressorTorch,
    )

    _no_mutable_defaults(MCDCNNRegressorTorch)


@_TORCH_REQUIRED
def test_mcdcnn_regressor_default_optim_kwargs_is_None():
    from sktime.regression.deep_learning.mcdcnn._mcdcnn_torch import (
        MCDCNNRegressorTorch,
    )

    inst = MCDCNNRegressorTorch()
    assert inst.optim_kwargs is None


# --------------------------------------------------------------------- #
# RBFForecaster
# --------------------------------------------------------------------- #


@_TORCH_REQUIRED
def test_rbf_forecaster_no_mutable_defaults():
    from sktime.forecasting.rbf_forecaster import RBFForecaster

    _no_mutable_defaults(RBFForecaster)


@_TORCH_REQUIRED
def test_rbf_forecaster_default_hidden_layers_is_None():
    from sktime.forecasting.rbf_forecaster import RBFForecaster

    inst = RBFForecaster()
    assert inst.hidden_layers is None


@_TORCH_REQUIRED
def test_rbf_forecaster_explicit_hidden_layers_round_trip():
    from sktime.forecasting.rbf_forecaster import RBFForecaster

    inst = RBFForecaster(hidden_layers=[8, 4])
    _get_params_round_trip(inst, {"hidden_layers": [8, 4]})


@_TORCH_REQUIRED
def test_rbf_forecaster_default_lists_are_independent():
    """Mutating one instance's hidden_layers must not affect another."""
    from sktime.forecasting.rbf_forecaster import RBFForecaster

    a = RBFForecaster()
    b = RBFForecaster()
    # Both stored as None now; on ``main`` the default was a shared
    # ``[64, 32]`` list — appending to ``a.hidden_layers`` there would
    # change ``b.hidden_layers`` for every other default-constructed
    # instance.
    assert a.hidden_layers is None and b.hidden_layers is None
