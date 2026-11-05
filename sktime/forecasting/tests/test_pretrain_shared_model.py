"""Tests that pretrain does not modify the model shared via a multiton.

Foundation model forecasters load their model through a multiton, so that all
instances with the same loading parameters share one model object. ``pretrain``
fine-tunes the model in place, and must therefore not act on the shared object.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import copy

import pytest

from sktime.forecasting.sundial import SundialForecaster
from sktime.forecasting.timemoe import TimeMoEForecaster
from sktime.forecasting.timesfm2 import TimesFM2Forecaster
from sktime.forecasting.ttm import TinyTimeMixerForecaster, _CachedTinyTimeMixer
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series
from sktime.utils.dependencies import _check_soft_dependencies

# index of the test parameter set that builds the model from a config,
# i.e., needs no download
_OFFLINE_PARAMS = {
    SundialForecaster: 0,
    TimeMoEForecaster: 0,
    TimesFM2Forecaster: 0,
    TinyTimeMixerForecaster: 1,
}

_FORECASTERS = [
    pytest.param(
        cls,
        marks=pytest.mark.skipif(
            not run_test_for_class(cls),
            reason="run test only if softdeps are present and incrementally",
        ),
    )
    for cls in _OFFLINE_PARAMS
]

FH = [1, 2, 3]


def _make_forecaster(cls, tmp_path):
    """Create a forecaster that trains visibly in a few steps, without download."""
    params = copy.deepcopy(cls.get_test_params()[_OFFLINE_PARAMS[cls]])
    training_args = dict(params.get("training_args") or {})
    training_args.update(
        {
            "output_dir": str(tmp_path),
            "max_steps": 3,
            "learning_rate": 1e-2,
            "report_to": "none",
            "use_cpu": True,
        }
    )
    # TimeMoE trainer args do not know ``use_cpu``
    if cls is TimeMoEForecaster:
        training_args.pop("use_cpu")
    params["training_args"] = training_args
    # no evaluation data is needed, and short test series cannot be split further
    params["validation_split"] = None
    return cls(**params)


def _get_shared_model(forecaster):
    """Get the multiton model of a forecaster, without fitting or pretraining it."""
    fresh = forecaster.clone()
    if isinstance(fresh, TinyTimeMixerForecaster):
        fresh._revision = fresh.revision
        return _CachedTinyTimeMixer(
            key=fresh._get_unique_key(),
            model_path=fresh.model_path,
            revision=fresh._revision,
            device=fresh.device,
            user_config=fresh._config,
            use_source_package=fresh.use_source_package,
            fit_strategy=fresh.fit_strategy,
        ).load()
    return fresh._load_model()


def _weights(model):
    """Snapshot of the weights of a model."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same_weights(weights_a, weights_b):
    """Whether two weight snapshots are identical."""
    import torch

    return weights_a.keys() == weights_b.keys() and all(
        torch.equal(weights_a[k], weights_b[k]) for k in weights_a
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="torch is required",
)
@pytest.mark.parametrize("cls", _FORECASTERS)
def test_pretrain_does_not_modify_shared_model(cls, tmp_path):
    """Pretraining one instance leaves the shared model and its clones alone."""
    y_panel = _make_hierarchical(
        hierarchy_levels=(4,), min_timepoints=60, max_timepoints=60, n_columns=1
    )

    forecaster = _make_forecaster(cls, tmp_path)
    shared = _get_shared_model(forecaster)
    weights_before = _weights(shared)

    pretrained = forecaster.clone().pretrain(y_panel, fh=FH)

    # pretraining must have an effect on the pretrained instance only
    assert pretrained.model_ is not shared
    assert not _same_weights(_weights(pretrained.model_), weights_before)
    assert _same_weights(_weights(shared), weights_before)

    # a clone of the pretrained instance keeps the pretrained weights,
    # without sharing the model with the instance or with the multiton
    cloned = pretrained.clone()
    assert cloned.model_ is not pretrained.model_
    assert cloned.model_ is not shared
    assert _same_weights(_weights(cloned.model_), _weights(pretrained.model_))

    # a new instance with the same parameters still gets the original weights
    assert _same_weights(_weights(_get_shared_model(forecaster)), weights_before)


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="torch is required",
)
@pytest.mark.parametrize("cls", _FORECASTERS)
def test_fit_after_pretrain_keeps_pretrained_model(cls, tmp_path):
    """Fit after pretrain continues from the pretrained model, not the shared one."""
    y = _make_series(n_timepoints=60)
    y_panel = _make_hierarchical(
        hierarchy_levels=(4,), min_timepoints=60, max_timepoints=60, n_columns=1
    )

    forecaster = _make_forecaster(cls, tmp_path)
    shared = _get_shared_model(forecaster)
    weights_before = _weights(shared)

    pretrained = forecaster.clone().pretrain(y_panel, fh=FH)
    pretrained_model = pretrained.model_
    pretrained_weights = _weights(pretrained_model)

    pretrained.fit(y, fh=FH)

    assert pretrained.model_ is not shared
    assert _same_weights(_weights(shared), weights_before)
    if cls is not TinyTimeMixerForecaster:
        # these forecasters do not train in fit, so the pretrained model is kept
        assert pretrained.model_ is pretrained_model
        assert _same_weights(_weights(pretrained.model_), pretrained_weights)
    else:
        # fit_strategy="full" trains on top of the pretrained model, in place
        assert pretrained.model_ is pretrained_model

    pretrained.predict()


@pytest.mark.skipif(
    not run_test_for_class(SundialForecaster),
    reason="run test only if softdeps are present and incrementally",
)
def test_zero_shot_predictions_unchanged_by_pretrain(tmp_path):
    """Predictions of a zero-shot instance do not change if another pretrains."""
    import torch

    y = _make_series(n_timepoints=60)
    y_panel = _make_hierarchical(
        hierarchy_levels=(4,), min_timepoints=60, max_timepoints=60, n_columns=1
    )

    forecaster = _make_forecaster(SundialForecaster, tmp_path)

    zero_shot = forecaster.clone().fit(y, fh=FH)
    torch.manual_seed(0)
    pred_before = zero_shot.predict()

    forecaster.clone().pretrain(y_panel)

    torch.manual_seed(0)
    assert zero_shot.predict().equals(pred_before)
