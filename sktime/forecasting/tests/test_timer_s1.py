"""Regression tests for the Timer-S1 forecaster."""

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.timer_s1 import TimerS1Forecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(TimerS1Forecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_random_init_model_is_in_eval_mode():
    """Randomly initialized model must be in eval mode so dropout is inactive.

    Regression test for #11310: ``_CachedTimerS1._load_randomly`` built the model
    from config without calling ``eval()``, so dropout stayed active and
    ``predict`` was non-deterministic unless ``deterministic=True`` seeded torch.
    """
    import torch

    config = {
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_attention_heads": 4,
        "num_experts": 4,
        "num_hidden_layers": 1,
        "num_mtp_tokens": 1,
        "dropout_rate": 0.1,
        "use_cache": False,
    }
    f = TimerS1Forecaster(model_path=None, config=config, deterministic=False)
    y = load_airline()
    f.fit(y, fh=[1, 2, 3])

    assert f.model_.training is False

    torch.manual_seed(0)
    y_pred_1 = f.predict()
    torch.manual_seed(1)
    y_pred_2 = f.predict()

    assert y_pred_1.equals(y_pred_2)
