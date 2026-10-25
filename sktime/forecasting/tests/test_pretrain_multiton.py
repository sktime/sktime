"""Regression tests for foundation-model pretrain and the multiton cache."""

import tempfile

import pytest

from sktime.forecasting.sundial import SundialForecaster
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series


def test_sundial_pretrain_does_not_mutate_shared_multiton():
    """Pretrain trains a private copy and leaves the cached module unchanged."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    params = SundialForecaster.get_test_params()[0]
    params["validation_split"] = None
    with tempfile.TemporaryDirectory() as output_dir:
        params["training_args"] = {
            "output_dir": output_dir,
            "max_steps": 1,
            "report_to": "none",
            "logging_strategy": "no",
            "save_strategy": "no",
            "disable_tqdm": True,
        }
        y = _make_series(n_columns=1, n_timepoints=24)
        y_panel = _make_hierarchical(
            hierarchy_levels=(3,),
            min_timepoints=24,
            max_timepoints=24,
            n_columns=1,
        )

        zero_shot = SundialForecaster(**params)
        zero_shot.fit(y, fh=[1])
        cached = zero_shot.model_
        cached_weight = next(cached.parameters()).detach().clone()

        pretrained = SundialForecaster(**params)
        pretrained.pretrain(y_panel, fh=[1])

        assert pretrained.model_ is not cached
        assert torch.equal(next(cached.parameters()).detach(), cached_weight)

        cloned = pretrained.clone()
        assert cloned.model_ is not cached
        assert cloned.model_ is not pretrained.model_
        pretrained_weight = next(pretrained.model_.parameters()).detach()
        cloned_weight = next(cloned.model_.parameters()).detach()
        assert torch.equal(cloned_weight, pretrained_weight)
