# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for ONNX export/import utilities."""

__author__ = ["ojuschugh1"]
__all__ = []

import pytest

from sktime.utils.dependencies import _check_soft_dependencies

onnx_available = _check_soft_dependencies(
    "skl2onnx", "onnx", "onnxruntime", severity="none"
)


def _fit_sklearn_clf(n_features=4):
    """Return a fitted LogisticRegression and its training data as float32.

    We use sklearn here because skl2onnx only converts sklearn estimators.
    """
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=50, n_features=n_features, random_state=42)
    X = X.astype(np.float32)
    return LogisticRegression().fit(X, y), X


# -- conversion tests (need the full onnx stack) ----------------------------


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_save_to_onnx_returns_model_proto():
    """path=None should return the in-memory ModelProto."""
    import onnx

    from sktime.utils.save import save_to_onnx

    clf, _ = _fit_sklearn_clf()
    assert isinstance(save_to_onnx(clf), onnx.ModelProto)


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_save_to_onnx_writes_file(tmp_path):
    """Giving a path should write a non-empty .onnx file."""
    from sktime.utils.save import save_to_onnx

    clf, _ = _fit_sklearn_clf()
    out = tmp_path / "model"
    assert save_to_onnx(clf, path=out) is None
    assert (tmp_path / "model.onnx").stat().st_size > 0


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_onnx_wrapper_predict():
    """Wrapper should return one prediction per input row."""
    from sktime.utils.save import save_to_onnx
    from sktime.utils.save._onnx import OnnxWrapper

    clf, X = _fit_sklearn_clf()
    wrapper = OnnxWrapper(save_to_onnx(clf).SerializeToString())
    assert len(wrapper.predict(X[:5])) == 5


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_onnx_wrapper_predict_proba():
    """predict_proba on a binary classifier should give (n, 2)."""
    from sktime.utils.save import save_to_onnx
    from sktime.utils.save._onnx import OnnxWrapper

    clf, X = _fit_sklearn_clf()
    wrapper = OnnxWrapper(save_to_onnx(clf).SerializeToString())
    assert wrapper.predict_proba(X[:5]).shape == (5, 2)


# -- load-path tests --------------------------------------------------------


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_load_from_onnx_bytes():
    """load_from_onnx should accept raw serialised bytes."""
    from sktime.utils.save import load_from_onnx, save_to_onnx

    clf, X = _fit_sklearn_clf()
    wrapper = load_from_onnx(save_to_onnx(clf).SerializeToString())
    assert len(wrapper.predict(X[:5])) == 5


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_load_from_onnx_model_proto():
    """load_from_onnx should also accept an onnx.ModelProto directly."""
    from sktime.utils.save import load_from_onnx, save_to_onnx

    clf, X = _fit_sklearn_clf()
    wrapper = load_from_onnx(save_to_onnx(clf))
    assert len(wrapper.predict(X[:5])) == 5


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_onnx_file_roundtrip(tmp_path):
    """Save -> load -> predict should work end-to-end via a file."""
    from sktime.utils.save import load_from_onnx, save_to_onnx

    clf, X = _fit_sklearn_clf()
    save_to_onnx(clf, path=tmp_path / "clf")
    wrapper = load_from_onnx(tmp_path / "clf.onnx")
    assert len(wrapper.predict(X[:5])) == 5


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_load_from_onnx_file_not_found():
    """A non-existent path should raise FileNotFoundError."""
    from sktime.utils.save import load_from_onnx

    with pytest.raises(FileNotFoundError):
        load_from_onnx("/no/such/file.onnx")


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_onnx_wrapper_repr():
    """repr should contain the class name."""
    from sktime.utils.save import save_to_onnx
    from sktime.utils.save._onnx import OnnxWrapper

    clf, _ = _fit_sklearn_clf()
    wrapper = OnnxWrapper(save_to_onnx(clf).SerializeToString())
    assert "OnnxWrapper" in repr(wrapper)


# -- error-path tests -------------------------------------------------------


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_predict_proba_raises_for_regressor():
    """Regression model has no proba output — should raise ValueError."""
    import numpy as np
    from sklearn.linear_model import LinearRegression

    from sktime.utils.save import save_to_onnx
    from sktime.utils.save._onnx import OnnxWrapper

    rng = np.random.RandomState(0)
    X = rng.rand(20, 4).astype(np.float32)
    y = rng.rand(20).astype(np.float32)
    reg = LinearRegression().fit(X, y)

    wrapper = OnnxWrapper(save_to_onnx(reg).SerializeToString())
    with pytest.raises(ValueError, match="probability output"):
        wrapper.predict_proba(X[:3])


@pytest.mark.skipif(not onnx_available, reason="onnx stack not installed")
def test_save_to_onnx_sktime_forecaster_fails():
    """Pure sktime forecaster isn't sklearn-compatible — should fail."""
    import pandas as pd
    from skl2onnx.common.data_types import FloatTensorType

    from sktime.forecasting.naive import NaiveForecaster
    from sktime.utils.save import save_to_onnx

    y = pd.Series(
        range(10),
        index=pd.period_range("2020", periods=10, freq="M"),
    )
    fh = NaiveForecaster(strategy="last")
    fh.fit(y)

    # bypass initial_types inference — the conversion itself should fail
    with pytest.raises(RuntimeError, match="sklearn-compatible"):
        save_to_onnx(fh, initial_types=[("X", FloatTensorType([None, 1]))])


def test_save_to_onnx_type_error():
    """Non-estimator objects should be rejected with TypeError."""
    from sktime.utils.save import save_to_onnx

    with pytest.raises(TypeError, match="fit"):
        save_to_onnx("just a string")


def test_save_to_onnx_not_fitted():
    """Unfitted sktime estimator should raise NotFittedError."""
    from sktime.exceptions import NotFittedError
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.utils.save import save_to_onnx

    with pytest.raises((NotFittedError, ImportError)):
        save_to_onnx(NaiveForecaster())
