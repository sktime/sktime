"""Tests for torch estimator constructor parameter naming conventions.

All ``torch`` based deep learning classifiers and regressors share the
canonical parameter names ``num_epochs``, ``optimizer`` and
``optimizer_kwargs`` from ``BaseDeepClassifierPytorch`` /
``BaseDeepRegressorTorch``. ``MCDCNNClassifierTorch`` and
``MCDCNNRegressorTorch`` historically used ``n_epochs``, ``optim`` and
``optim_kwargs`` instead; those names are kept as deprecated aliases.
"""

import inspect

import pytest

from sktime.classification.deep_learning.base import BaseDeepClassifierPytorch
from sktime.registry import all_estimators
from sktime.regression.deep_learning.base import BaseDeepRegressorTorch

TORCH_ESTIMATORS = [
    est
    for est in all_estimators(
        estimator_types=["classifier", "regressor"], return_names=False
    )
    if issubclass(est, (BaseDeepClassifierPytorch, BaseDeepRegressorTorch))
]

CANONICAL_PARAMS = {"num_epochs", "optimizer", "optimizer_kwargs"}
LEGACY_PARAMS = {"n_epochs", "optim", "optim_kwargs"}


def _init_params(est):
    return set(inspect.signature(est.__init__).parameters) - {"self"}


@pytest.mark.skipif(
    len(TORCH_ESTIMATORS) == 0,
    reason="torch estimators are only available if torch is installed",
)
@pytest.mark.parametrize("est", TORCH_ESTIMATORS, ids=lambda e: e.__name__)
def test_torch_estimators_use_canonical_param_names(est):
    """All torch estimators expose the canonical base class parameter names."""
    params = _init_params(est)
    missing = CANONICAL_PARAMS - params
    assert not missing, (
        f"{est.__name__} does not accept the canonical torch base class "
        f"parameter(s) {sorted(missing)} used by every other torch estimator."
    )


@pytest.mark.skipif(
    len(TORCH_ESTIMATORS) == 0,
    reason="torch estimators are only available if torch is installed",
)
@pytest.mark.parametrize("est", TORCH_ESTIMATORS, ids=lambda e: e.__name__)
def test_torch_estimators_no_undeprecated_legacy_names(est):
    """Legacy MCDCNN parameter names are only kept as deprecated aliases."""
    from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifierTorch
    from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressorTorch

    deprecated_alias_hosts = {MCDCNNClassifierTorch, MCDCNNRegressorTorch}
    if est in deprecated_alias_hosts:
        pytest.skip("MCDCNN torch estimators keep legacy names as deprecated aliases")
    legacy = _init_params(est) & LEGACY_PARAMS
    assert not legacy, (
        f"{est.__name__} uses legacy parameter name(s) {sorted(legacy)}; "
        "torch estimators should use `num_epochs`, `optimizer` and "
        "`optimizer_kwargs`."
    )
