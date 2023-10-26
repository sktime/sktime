"""Arsenal test code."""
import pytest

from sktime.classification.kernel_based import Arsenal
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class


# A reference to this issue is also present inside sktime/tests/_config.py,
# and needs to be removed from `EXCLUDED_TESTS` upon resolution.
@pytest.mark.skip(
    reason="Fails because of `len(obj.estimators_)==1`, "
    "refer issue #5488 for details."
)
@pytest.mark.skipif(
    not run_test_for_class(Arsenal),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_contracted_arsenal():
    """Test of contracted Arsenal on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted Arsenal
    arsenal = Arsenal(
        time_limit_in_minutes=0.25,
        contract_max_n_estimators=5,
        num_kernels=20,
        random_state=0,
    )
    arsenal.fit(X_train, y_train)

    assert len(arsenal.estimators_) > 1
