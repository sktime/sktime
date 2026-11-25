"""Tests for MCDCNN torch estimator parameter naming and deprecated aliases."""

import warnings

import pytest

from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifierTorch
from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressorTorch
from sktime.tests.test_switch import run_test_for_class

ESTIMATORS = [MCDCNNClassifierTorch, MCDCNNRegressorTorch]


@pytest.mark.skipif(
    not run_test_for_class(ESTIMATORS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("est_cls", ESTIMATORS, ids=lambda c: c.__name__)
def test_canonical_params_stored_and_mapped(est_cls):
    """Canonical parameter names are stored and reach the torch base class."""
    est = est_cls(
        num_epochs=5, optimizer="Adam", optimizer_kwargs={"weight_decay": 0.1}
    )
    assert est.num_epochs == 5
    assert est.optimizer == "Adam"
    assert est.optimizer_kwargs == {"weight_decay": 0.1}
    # deprecated aliases stay unset
    assert est.n_epochs == "deprecated"
    assert est.optim == "deprecated"
    assert est.optim_kwargs == "deprecated"


@pytest.mark.skipif(
    not run_test_for_class(ESTIMATORS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("est_cls", ESTIMATORS, ids=lambda c: c.__name__)
def test_default_optimizer_resolution(est_cls):
    """Legacy default optimizer resolution is applied at fit time."""
    est = est_cls()
    # constructor stores the canonical parameters verbatim (estimator contract)
    assert est.optimizer is None
    assert est.optimizer_kwargs is None
    # the historical MCDCNN default is resolved at fit time
    optimizer, optimizer_kwargs = est._resolve_optimizer()
    assert optimizer == "SGD"
    assert optimizer_kwargs == {"momentum": 0.9, "weight_decay": 0.0005}


@pytest.mark.skipif(
    not run_test_for_class(ESTIMATORS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("est_cls", ESTIMATORS, ids=lambda c: c.__name__)
def test_deprecated_aliases_warn_and_map(est_cls):
    """Deprecated aliases warn and are resolved to the canonical parameters."""
    with pytest.warns(DeprecationWarning, match=r"'n_epochs' will be renamed"):
        est = est_cls(n_epochs=7)
    # both names are stored verbatim; the alias wins for training
    assert est.num_epochs == 120
    assert est.n_epochs == 7
    assert est._num_epochs == 7

    with pytest.warns(DeprecationWarning, match=r"'optim' will be renamed"):
        est = est_cls(optim="Adam")
    assert est.optim == "Adam"
    # canonical parameter stays verbatim; the alias is resolved at fit time
    assert est.optimizer is None
    assert est._resolve_optimizer() == ("Adam", None)

    with pytest.warns(DeprecationWarning, match=r"'optim_kwargs' will be renamed"):
        est = est_cls(optim_kwargs={"weight_decay": 0.2})
    assert est.optim_kwargs == {"weight_decay": 0.2}
    assert est.optimizer_kwargs is None
    assert est._resolve_optimizer() == ("SGD", {"weight_decay": 0.2})


@pytest.mark.skipif(
    not run_test_for_class(ESTIMATORS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("est_cls", ESTIMATORS, ids=lambda c: c.__name__)
def test_deprecated_alias_takes_precedence(est_cls):
    """A set deprecated alias takes precedence, warning, per the rename recipe."""
    with pytest.warns(DeprecationWarning):
        est = est_cls(num_epochs=3, n_epochs=7)
    # both stored verbatim; the alias supplies the value used for training
    assert est.num_epochs == 3
    assert est.n_epochs == 7
    assert est._num_epochs == 7

    with pytest.warns(DeprecationWarning):
        est = est_cls(optimizer="Adam", optim="SGD")
    assert est._resolve_optimizer()[0] == "SGD"

    with pytest.warns(DeprecationWarning):
        est = est_cls(optimizer_kwargs={"a": 1}, optim_kwargs={"b": 2})
    assert est._resolve_optimizer()[1] == {"b": 2}

    # equal values (clone/reset round-trip) are accepted, still warning that
    # the deprecated aliases were used
    with pytest.warns(DeprecationWarning):
        est = est_cls(num_epochs=7, n_epochs=7, optimizer="Adam", optim="Adam")
    assert est.num_epochs == 7
    assert est._num_epochs == 7
    assert est._resolve_optimizer()[0] == "Adam"


@pytest.mark.skipif(
    not run_test_for_class(ESTIMATORS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("est_cls", ESTIMATORS, ids=lambda c: c.__name__)
def test_no_warning_with_canonical_params_only(est_cls):
    """Canonical construction does not emit deprecation warnings."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        est_cls()
        est_cls(num_epochs=2, optimizer="Adam", optimizer_kwargs={"eps": 1e-6})
    future = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert not future, [str(w.message) for w in future]
