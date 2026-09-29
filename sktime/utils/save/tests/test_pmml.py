# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for PMML export/import utilities."""

__author__ = ["ojuschugh1"]
__all__ = []

import pytest

from sktime.utils.dependencies import _check_soft_dependencies

sklearn2pmml_available = _check_soft_dependencies("sklearn2pmml", severity="none")
pypmml_available = _check_soft_dependencies("pypmml", severity="none")


def _fit_pmml_pipeline():
    """Return a fitted PMMLPipeline and its DataFrame.

    sklearn2pmml requires a PMMLPipeline wrapping an sklearn estimator,
    and columns must be named so the PMML model can map them.
    """
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn2pmml.pipeline import PMMLPipeline

    X_arr, y = make_classification(n_samples=50, n_features=4, random_state=42)
    cols = [f"x{i}" for i in range(X_arr.shape[1])]
    X_df = pd.DataFrame(X_arr, columns=cols)
    pipeline = PMMLPipeline([("clf", LogisticRegression())])
    pipeline.fit(X_df, y)
    return pipeline, X_df


# -- input-validation tests (no optional deps needed) -----------------------


def test_save_to_pmml_bad_backend():
    """Unknown backend name should raise ValueError."""
    from sktime.datasets import load_airline
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.utils.save import save_to_pmml

    y = load_airline()
    fh = NaiveForecaster()
    fh.fit(y, fh=[1, 2, 3])

    with pytest.raises(ValueError, match="pmml_backend"):
        save_to_pmml(fh, path="out.pmml", pmml_backend="bogus")


def test_save_to_pmml_type_error():
    """Non-estimator objects should be rejected."""
    from sktime.utils.save import save_to_pmml

    with pytest.raises(TypeError, match="estimator"):
        save_to_pmml("not_an_estimator", path="out.pmml")


def test_save_to_pmml_not_fitted():
    """Unfitted sktime estimator should raise NotFittedError."""
    from sktime.exceptions import NotFittedError
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.utils.save import save_to_pmml

    with pytest.raises((NotFittedError, ImportError)):
        save_to_pmml(NaiveForecaster(), path="out.pmml")


def test_load_from_pmml_file_not_found():
    """Missing file should raise FileNotFoundError."""
    from sktime.utils.save import load_from_pmml

    with pytest.raises((FileNotFoundError, ImportError)):
        load_from_pmml("/no/such/model.pmml")


# -- roundtrip / wrapper tests (need sklearn2pmml and/or pypmml) ------------


@pytest.mark.skipif(not sklearn2pmml_available, reason="sklearn2pmml not installed")
def test_sklearn2pmml_roundtrip(tmp_path):
    """Export via sklearn2pmml and check the .pmml file was created."""
    from sktime.utils.save import save_to_pmml

    pipeline, _ = _fit_pmml_pipeline()
    save_to_pmml(pipeline, path=tmp_path / "model", pmml_backend="sklearn2pmml")
    assert (tmp_path / "model.pmml").stat().st_size > 0


@pytest.mark.skipif(not pypmml_available, reason="pypmml not installed")
def test_pmml_wrapper_predict(tmp_path):
    """PmmlWrapper.predict should return one value per input row."""
    if not sklearn2pmml_available:
        pytest.skip("sklearn2pmml needed to create the test fixture")

    from sktime.utils.save import save_to_pmml
    from sktime.utils.save._pmml import PmmlWrapper

    pipeline, X_df = _fit_pmml_pipeline()
    save_to_pmml(pipeline, path=tmp_path / "model", pmml_backend="sklearn2pmml")

    wrapper = PmmlWrapper(str(tmp_path / "model.pmml"))
    assert len(wrapper.predict(X_df.iloc[:5])) == 5


@pytest.mark.skipif(not pypmml_available, reason="pypmml not installed")
def test_pmml_wrapper_rejects_ndarray(tmp_path):
    """numpy arrays should be rejected — PMML needs named fields."""
    if not sklearn2pmml_available:
        pytest.skip("sklearn2pmml needed to create the test fixture")

    import numpy as np

    from sktime.utils.save import save_to_pmml
    from sktime.utils.save._pmml import PmmlWrapper

    pipeline, X_df = _fit_pmml_pipeline()
    save_to_pmml(pipeline, path=tmp_path / "model", pmml_backend="sklearn2pmml")

    wrapper = PmmlWrapper(str(tmp_path / "model.pmml"))
    with pytest.raises(TypeError, match="pd.DataFrame"):
        wrapper.predict(np.array(X_df.iloc[:5]))


@pytest.mark.skipif(not pypmml_available, reason="pypmml not installed")
def test_pmml_wrapper_model_info(tmp_path):
    """get_model_info should return a dict with the expected keys."""
    if not sklearn2pmml_available:
        pytest.skip("sklearn2pmml needed to create the test fixture")

    from sktime.utils.save import save_to_pmml
    from sktime.utils.save._pmml import PmmlWrapper

    pipeline, _ = _fit_pmml_pipeline()
    save_to_pmml(pipeline, path=tmp_path / "model", pmml_backend="sklearn2pmml")

    info = PmmlWrapper(str(tmp_path / "model.pmml")).get_model_info()
    assert "model_class" in info and "path" in info


@pytest.mark.skipif(not pypmml_available, reason="pypmml not installed")
def test_pmml_wrapper_repr(tmp_path):
    """repr should mention the class name."""
    if not sklearn2pmml_available:
        pytest.skip("sklearn2pmml needed to create the test fixture")

    from sktime.utils.save import save_to_pmml
    from sktime.utils.save._pmml import PmmlWrapper

    pipeline, _ = _fit_pmml_pipeline()
    save_to_pmml(pipeline, path=tmp_path / "model", pmml_backend="sklearn2pmml")

    assert "PmmlWrapper" in repr(PmmlWrapper(str(tmp_path / "model.pmml")))
