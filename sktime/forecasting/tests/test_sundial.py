"""Regression tests for the Sundial forecaster."""

import pytest

from sktime.forecasting.sundial import SundialForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series

pytestmark = pytest.mark.skipif(
    not run_test_for_class(SundialForecaster),
    reason="run test only if torch and transformers are installed",
)


def test_sundial_pretrain_does_not_mutate_zero_shot_model(tmp_path):
    """Pretraining one Sundial instance does not change another instance."""
    import torch

    params = SundialForecaster.get_test_params()[0]
    params["training_args"] = {
        "output_dir": tmp_path,
        "max_steps": 5,
        "learning_rate": 1e-2,
        "report_to": "none",
        "use_cpu": True,
    }

    y = _make_series(n_timepoints=60)
    y_panel = _make_hierarchical(
        hierarchy_levels=(4,),
        min_timepoints=60,
        max_timepoints=60,
        n_columns=1,
    )

    zero_shot = SundialForecaster(**params).fit(y, fh=[1, 2, 3])
    torch.manual_seed(0)
    pred_before = zero_shot.predict()

    pretrained = SundialForecaster(**params).pretrain(y_panel)

    torch.manual_seed(0)
    assert zero_shot.predict().equals(pred_before)
    assert pretrained.model_ is not zero_shot.model_
