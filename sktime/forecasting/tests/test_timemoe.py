"""Regression tests for the TimeMoE forecaster."""

import pytest

from sktime.forecasting.timemoe import TimeMoEForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series

pytestmark = pytest.mark.skipif(
    not run_test_for_class(TimeMoEForecaster),
    reason="run test only if torch, transformers, and accelerate are installed",
)


def test_timemoe_pretrain_does_not_mutate_zero_shot_model(tmp_path):
    """Pretraining TimeMoE does not change a separate zero-shot instance."""
    import torch

    params = TimeMoEForecaster.get_test_params()[0]
    params["training_args"] = {
        "max_steps": 5,
        "learning_rate": 1e-2,
        "per_device_train_batch_size": 1,
        "output_dir": tmp_path,
        "report_to": [],
        "logging_strategy": "no",
        "save_strategy": "no",
    }

    y = _make_series(n_timepoints=60)
    y_panel = _make_hierarchical(
        hierarchy_levels=(4,),
        min_timepoints=60,
        max_timepoints=60,
        n_columns=1,
    )

    zero_shot = TimeMoEForecaster(**params).fit(y, fh=[1, 2, 3])
    torch.manual_seed(0)
    pred_before = zero_shot.predict()

    pretrained = TimeMoEForecaster(**params).pretrain(y_panel)

    torch.manual_seed(0)
    assert zero_shot.predict().equals(pred_before)
    assert pretrained.model_ is not zero_shot.model_

    clone = pretrained.clone()
    assert clone.model_ is pretrained.model_

    pretrained = pretrained.fit(y, fh=[1, 2, 3])
    clone = clone.fit(y, fh=[1, 2, 3])
    torch.manual_seed(0)
    parent_before = pretrained.predict()
    clone.pretrain(y_panel)
    torch.manual_seed(0)
    assert pretrained.predict().equals(parent_before)
