"""HIVE-COTE v2 test code."""

import pytest

from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.sklearn import RotationForest
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(HIVECOTEV2),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_contracted_hivecote_v2():
    """Test of contracted HIVECOTEV2 on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted HIVE-COTE v2
    hc2 = HIVECOTEV2(
        stc_params={
            "estimator": RotationForest(contract_max_n_estimators=1),
            "contract_max_n_shapelet_samples": 5,
            "max_shapelets": 5,
            "batch_size": 5,
        },
        drcif_params={
            "contract_max_n_estimators": 1,
            "n_intervals": 2,
            "att_subsample_size": 2,
        },
        arsenal_params={"num_kernels": 5, "contract_max_n_estimators": 1},
        tde_params={
            "contract_max_n_parameter_samples": 1,
            "max_ensemble_size": 1,
            "randomly_selected_params": 1,
        },
        time_limit_in_minutes=0.25,
        random_state=0,
    )
    hc2.fit(X_train, y_train)
