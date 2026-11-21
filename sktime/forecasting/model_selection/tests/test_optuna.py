# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for Optuna sampler dependencies (issue #8714)."""

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.model_selection import ForecastingOptunaSearchCV
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import SingleWindowSplitter


@pytest.mark.parametrize("sampler_name", [None, "RandomSampler", "CmaEsSampler"])
def test_optuna_sampler_dependencies(sampler_name, monkeypatch):
    """Only CMA-ES requires cmaes, and missing it produces installation guidance."""
    optuna = pytest.importorskip("optuna")
    from skbase.utils.dependencies import _dependencies

    get_version = _dependencies._get_pkg_version

    def without_cmaes(package, *args, **kwargs):
        if package == "cmaes":
            return None
        return get_version(package, *args, **kwargs)

    monkeypatch.setattr(_dependencies, "_get_pkg_version", without_cmaes)
    sampler = getattr(optuna.samplers, sampler_name)(seed=42) if sampler_name else None
    tuner = ForecastingOptunaSearchCV(
        NaiveForecaster(),
        cv=SingleWindowSplitter(fh=[1]),
        param_grid={"window_length": optuna.distributions.IntDistribution(2, 5)},
        sampler=sampler,
        n_evals=3,
        backend=None,
    )
    y = pd.Series(np.arange(24, dtype=float) + 1)
    if sampler_name == "CmaEsSampler":
        assert tuner.get_tag("python_dependencies") == ["optuna", "cmaes"]
        with pytest.raises(ModuleNotFoundError, match="pip install cmaes"):
            tuner.fit(y)
    else:
        assert tuner.get_tag("python_dependencies") == ["optuna"]
        tuner.fit(y)
        assert len(tuner.cv_results_) == 3


def test_optuna_cmaes_single_parameter():
    """Exercise CMA-ES beyond startup on a one-dimensional numerical search."""
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("cmaes")
    tuner = ForecastingOptunaSearchCV(
        NaiveForecaster(),
        cv=SingleWindowSplitter(fh=[1]),
        param_grid={"window_length": optuna.distributions.IntDistribution(2, 5)},
        sampler=optuna.samplers.CmaEsSampler(seed=42),
        n_evals=4,
        backend=None,
    )
    y = pd.Series(np.arange(24, dtype=float) + 1)
    tuner.fit(y)
    assert len(tuner.cv_results_) == 4
    assert np.isfinite(tuner.best_score_)
    assert np.isfinite(tuner.predict(fh=[1])).all()
