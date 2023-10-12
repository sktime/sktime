"""Tests for mlflow-sktime custom model flavor."""

__author__ = ["benjaminbluhm"]

import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_airline, load_longley
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.multiindex import flatten_multiindex
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("mlflow", "boto3", "moto", "botocore", severity="none"):
    import boto3
    import moto
    from botocore.config import Config

if not sys.platform.startswith("linux"):
    pytest.skip("Skipping MLflow tests for Windows and macOS", allow_module_level=True)


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary path to save/log model."""
    return tmp_path.joinpath("model")


@pytest.fixture(scope="module")
def mock_s3_bucket():
    """Create a mock S3 bucket using moto.

    Returns
    -------
    string with name of mock S3 bucket
    """
    with moto.mock_s3():
        bucket_name = "mock-bucket"
        my_config = Config(region_name="us-east-1")
        s3_client = boto3.client("s3", config=my_config)
        s3_client.create_bucket(Bucket=bucket_name)
        yield bucket_name


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.fixture
def sktime_custom_env(tmp_path):
    """Create a conda environment and returns path to conda environment yml file."""
    from mlflow.utils.environment import _mlflow_conda_env

    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
    return conda_env


@pytest.fixture(scope="module")
def test_data_airline():
    """Create sample data for univariate model without exogenous regressor."""
    return load_airline()


@pytest.fixture(scope="module")
def test_data_longley():
    """Create sample data for univariate model with exogenous regressor."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    return y_train, y_test, X_train, X_test


@pytest.fixture(scope="module")
def auto_arima_model(test_data_airline):
    """Create instance of fitted auto arima model."""
    return AutoARIMA(sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True).fit(
        test_data_airline, fh=[1, 2, 3]
    )


@pytest.fixture(scope="module")
def naive_forecaster_model_with_regressor(test_data_longley):
    """Create instance of fitted naive forecaster model."""
    y_train, _, X_train, _ = test_data_longley
    model = NaiveForecaster()
    return model.fit(y_train, X_train, fh=[1, 2, 3])


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_auto_arima_model_save_and_load(
    auto_arima_model, model_path, serialization_format
):
    """Test saving and loading of native sktime auto_arima_model."""
    from sktime.utils import mlflow_sktime

    mlflow_sktime.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = mlflow_sktime.load_model(
        model_uri=model_path,
    )

    np.testing.assert_array_equal(auto_arima_model.predict(), loaded_model.predict())


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_auto_arima_model_pyfunc_output(
    auto_arima_model, model_path, serialization_format
):
    """Test auto arima prediction of loaded pyfunc model."""
    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = {
        "predict_method": [
            "predict",
            "predict_interval",
            "predict_quantiles",
            "predict_var",
        ]
    }
    mlflow_sktime.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    model_predict = auto_arima_model.predict()
    model_predict_interval = auto_arima_model.predict_interval()
    model_predict_interval.columns = flatten_multiindex(model_predict_interval)
    model_predict_quantiles = auto_arima_model.predict_quantiles()
    model_predict_quantiles.columns = flatten_multiindex(model_predict_quantiles)
    model_predict_var = auto_arima_model.predict_var()
    model_predictions = pd.concat(
        [
            model_predict,
            model_predict_interval,
            model_predict_quantiles,
            model_predict_var,
        ],
        axis=1,
        keys=[
            "model_predict",
            "model_predict_interval",
            "model_predict_quantiles",
            "model_predict_var",
        ],
    )

    pyfunc_predict = loaded_pyfunc.predict(pd.DataFrame())
    np.testing.assert_array_equal(model_predictions.values, pyfunc_predict.values)


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_auto_arima_model_pyfunc_with_params_output(auto_arima_model, model_path):
    """Test auto arima prediction of loaded pyfunc model with parameters."""
    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = {
        "predict_method": {
            "predict": {},
            "predict_interval": {"coverage": [0.1, 0.9]},
            "predict_proba": {"quantiles": [0.1, 0.9], "marginal": True},
            "predict_quantiles": {"alpha": [0.1, 0.9]},
            "predict_var": {"cov": True},
        }
    }
    mlflow_sktime.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
    )
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    model_predict = auto_arima_model.predict()
    model_predict_interval = auto_arima_model.predict_interval(coverage=[0.1, 0.9])
    model_predict_interval.columns = flatten_multiindex(model_predict_interval)
    model_predict_proba_dist = auto_arima_model.predict_proba()
    model_predict_proba = model_predict_proba_dist.quantile([0.1, 0.9])
    model_predict_proba.columns = flatten_multiindex(model_predict_proba)
    model_predict_quantiles = auto_arima_model.predict_quantiles(alpha=[0.1, 0.9])
    model_predict_quantiles.columns = flatten_multiindex(model_predict_quantiles)
    model_predict_var = auto_arima_model.predict_var(cov=True)
    model_predictions = pd.concat(
        [
            model_predict,
            model_predict_interval,
            model_predict_proba,
            model_predict_quantiles,
            model_predict_var,
        ],
        axis=1,
        keys=[
            "model_predict",
            "model_predict_interval",
            "model_predict_proba",
            "model_predict_quantiles",
            "model_predict_var",
        ],
    )

    pyfunc_predict = loaded_pyfunc.predict(pd.DataFrame())
    np.testing.assert_array_equal(model_predictions.values, pyfunc_predict.values)


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_auto_arima_model_pyfunc_without_params_output(auto_arima_model, model_path):
    """Test auto arima prediction of loaded pyfunc model without parameters."""
    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = {
        "predict_method": {
            "predict": {},
            "predict_interval": {},
            "predict_proba": {"quantiles": [0.1, 0.9]},
            "predict_quantiles": {},
            "predict_var": {},
        }
    }
    mlflow_sktime.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
    )
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    model_predict = auto_arima_model.predict()
    model_predict_interval = auto_arima_model.predict_interval()
    model_predict_interval.columns = flatten_multiindex(model_predict_interval)
    model_predict_proba_dist = auto_arima_model.predict_proba()
    model_predict_proba = model_predict_proba_dist.quantile([0.1, 0.9])
    model_predict_proba.columns = flatten_multiindex(model_predict_proba)
    model_predict_quantiles = auto_arima_model.predict_quantiles()
    model_predict_quantiles.columns = flatten_multiindex(model_predict_quantiles)
    model_predict_var = auto_arima_model.predict_var()
    model_predictions = pd.concat(
        [
            model_predict,
            model_predict_interval,
            model_predict_proba,
            model_predict_quantiles,
            model_predict_var,
        ],
        axis=1,
        keys=[
            "model_predict",
            "model_predict_interval",
            "model_predict_proba",
            "model_predict_quantiles",
            "model_predict_var",
        ],
    )

    pyfunc_predict = loaded_pyfunc.predict(pd.DataFrame())
    np.testing.assert_array_equal(model_predictions.values, pyfunc_predict.values)


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_auto_arima_model_pyfunc_without_conf_output(auto_arima_model, model_path):
    """Test auto arima prediction of loaded pyfunc model without config."""
    from sktime.utils import mlflow_sktime

    delattr(auto_arima_model, "pyfunc_predict_conf") if hasattr(
        auto_arima_model, "pyfunc_predict_conf"
    ) else None
    mlflow_sktime.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
    )
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    model_predict = auto_arima_model.predict()

    pyfunc_predict = loaded_pyfunc.predict(pd.DataFrame())
    np.testing.assert_array_equal(model_predict.values, pyfunc_predict.values)


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_naive_forecaster_model_with_regressor_pyfunc_output(
    naive_forecaster_model_with_regressor, model_path, test_data_longley
):
    """Test naive forecaster prediction of loaded pyfunc model."""
    from sktime.utils import mlflow_sktime

    _, _, _, X_test = test_data_longley

    naive_forecaster_model_with_regressor.pyfunc_predict_conf = {
        "predict_method": [
            "predict",
            "predict_interval",
            "predict_quantiles",
            "predict_var",
        ]
    }
    mlflow_sktime.save_model(
        sktime_model=naive_forecaster_model_with_regressor, path=model_path
    )
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    model_predict = naive_forecaster_model_with_regressor.predict(X=X_test)
    model_predict_interval = naive_forecaster_model_with_regressor.predict_interval(
        X=X_test
    )
    model_predict_interval.columns = flatten_multiindex(model_predict_interval)
    model_predict_quantiles = naive_forecaster_model_with_regressor.predict_quantiles(
        X=X_test
    )
    model_predict_quantiles.columns = flatten_multiindex(model_predict_quantiles)
    model_predict_var = naive_forecaster_model_with_regressor.predict_var(X=X_test)
    model_predictions = pd.concat(
        [
            model_predict,
            model_predict_interval,
            model_predict_quantiles,
            model_predict_var,
        ],
        axis=1,
        keys=[
            "model_predict",
            "model_predict_interval",
            "model_predict_quantiles",
            "model_predict_var",
        ],
    )

    pyfunc_predict = loaded_pyfunc.predict(X_test)
    np.testing.assert_array_equal(model_predictions.values, pyfunc_predict.values)


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("use_signature", [True, False])
@pytest.mark.parametrize("use_example", [True, False])
def test_signature_and_examples_saved_correctly(
    auto_arima_model, test_data_airline, model_path, use_signature, use_example
):
    """Test saving of mlflow signature and example for native sktime predict method."""
    from mlflow.models import Model, infer_signature
    from mlflow.models.utils import _read_example

    from sktime.utils import mlflow_sktime

    # Note: Signature inference fails on native model predict_interval/predict_quantiles
    prediction = auto_arima_model.predict()
    signature = (
        infer_signature(test_data_airline, prediction) if use_signature else None
    )
    example = (
        pd.DataFrame(test_data_airline[0:5].copy(deep=False)) if use_example else None
    )
    mlflow_sktime.save_model(
        auto_arima_model, path=model_path, signature=signature, input_example=example
    )
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path).copy(deep=False)
        np.testing.assert_array_equal(r_example, example)


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("use_signature", [True, False])
def test_predict_var_signature_saved_correctly(
    auto_arima_model, test_data_airline, model_path, use_signature
):
    """Test saving of mlflow signature for native sktime predict_var method."""
    from mlflow.models import Model, infer_signature

    from sktime.utils import mlflow_sktime

    # Note: Signature inference fails on native model predict_interval/predict_quantiles
    prediction = auto_arima_model.predict_var()
    signature = (
        infer_signature(test_data_airline, prediction) if use_signature else None
    )
    mlflow_sktime.save_model(auto_arima_model, path=model_path, signature=signature)
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("use_signature", [True, False])
@pytest.mark.parametrize("use_example", [True, False])
def test_signature_and_example_for_pyfunc_predict(
    auto_arima_model, model_path, test_data_airline, use_signature, use_example
):
    """Test saving of mlflow signature and example for pyfunc predict."""
    from mlflow.models import Model, infer_signature
    from mlflow.models.utils import _read_example

    from sktime.utils import mlflow_sktime

    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")
    auto_arima_model.pyfunc_predict_conf = {
        "predict_method": {
            "predict": None,
            "predict_interval": {"coverage": [0.1, 0.9]},
            "predict_quantiles": {"alpha": [0.1, 0.9]},
            "predict_var": {"cov": False},
        }
    }
    mlflow_sktime.save_model(sktime_model=auto_arima_model, path=model_path_primary)
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path_primary)

    forecast = loaded_pyfunc.predict(pd.DataFrame())
    signature = infer_signature(test_data_airline, forecast) if use_signature else None
    example = (
        pd.DataFrame(test_data_airline[0:5].copy(deep=False)) if use_example else None
    )
    mlflow_sktime.save_model(
        auto_arima_model,
        path=model_path_secondary,
        signature=signature,
        input_example=example,
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path_secondary).copy(deep=False)
        np.testing.assert_array_equal(r_example, example)


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_load_from_remote_uri_succeeds(auto_arima_model, model_path, mock_s3_bucket):
    """Test loading native sktime model from mock S3 bucket."""
    from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository

    from sktime.utils import mlflow_sktime

    mlflow_sktime.save_model(sktime_model=auto_arima_model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = os.path.join(artifact_root, artifact_path)
    reloaded_sktime_model = mlflow_sktime.load_model(model_uri=model_uri)

    np.testing.assert_array_equal(
        auto_arima_model.predict(),
        reloaded_sktime_model.predict(),
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("should_start_run", [True, False])
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_log_model(auto_arima_model, tmp_path, should_start_run, serialization_format):
    """Test logging and reloading sktime model."""
    import mlflow
    from mlflow import pyfunc
    from mlflow.models import Model
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
    from mlflow.utils.environment import _mlflow_conda_env

    from sktime.utils import mlflow_sktime

    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "sktime"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
        model_info = mlflow_sktime.log_model(
            sktime_model=auto_arima_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            serialization_format=serialization_format,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = mlflow_sktime.load_model(
            model_uri=model_uri,
        )
        np.testing.assert_array_equal(
            auto_arima_model.predict(), reloaded_model.predict()
        )
        model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
    finally:
        mlflow.end_run()


@pytest.mark.xfail(reason="known failure to be debugged, see #4904")
@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_log_model_calls_register_model(auto_arima_model, tmp_path):
    """Test log model calls register model."""
    import mlflow
    from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
    from mlflow.utils.environment import _mlflow_conda_env

    from sktime.utils import mlflow_sktime

    artifact_path = "sktime"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
        mlflow_sktime.log_model(
            sktime_model=auto_arima_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            registered_model_name="SktimeModel",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri,
            "SktimeModel",
            await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        )


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_log_model_no_registered_model_name(auto_arima_model, tmp_path):
    """Test log model calls register model without registered model name."""
    import mlflow
    from mlflow.utils.environment import _mlflow_conda_env

    from sktime.utils import mlflow_sktime

    artifact_path = "sktime"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
        mlflow_sktime.log_model(
            sktime_model=auto_arima_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.register_model.assert_not_called()


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_pyfunc_raises_invalid_attribute_type(auto_arima_model, model_path):
    """Test pyfunc raises exception with invalid attribute type."""
    from mlflow.exceptions import MlflowException

    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = ["predict"]
    mlflow_sktime.save_model(sktime_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match=f"Attribute {mlflow_sktime.PYFUNC_PREDICT_CONF} must be of type dict.",
    ):
        loaded_pyfunc.predict(pd.DataFrame())


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_pyfunc_raises_invalid_dict_key(auto_arima_model, model_path):
    """Test pyfunc raises exception with invalid dict key."""
    from mlflow.exceptions import MlflowException

    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = {"prediction_method": ["predict"]}
    mlflow_sktime.save_model(sktime_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match=f"Attribute {mlflow_sktime.PYFUNC_PREDICT_CONF} must contain ",
    ):
        loaded_pyfunc.predict(pd.DataFrame())


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_pyfunc_raises_invalid_dict_value_type(auto_arima_model, model_path):
    """Test pyfunc raises exception with invalid dict value type."""
    from mlflow.exceptions import MlflowException

    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = {"predict_method": "predict"}
    mlflow_sktime.save_model(sktime_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException, match="Dictionary value must be of type dict or list"
    ):
        loaded_pyfunc.predict(pd.DataFrame())


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_pyfunc_raises_invalid_dict_value(auto_arima_model, model_path):
    """Test pyfunc raises exception with invalid dict value."""
    from mlflow.exceptions import MlflowException

    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = {"predict_method": ["forecast"]}
    mlflow_sktime.save_model(sktime_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match=f"The provided {mlflow_sktime.PYFUNC_PREDICT_CONF_KEY} values must be ",
    ):
        loaded_pyfunc.predict(pd.DataFrame())


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_pyfunc_predict_proba_raises_invalid_attribute_type(
    auto_arima_model, model_path
):
    """Test pyfunc predict_proba raises exception with invalid attribute type."""
    from mlflow.exceptions import MlflowException

    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = {"predict_method": ["predict_proba"]}
    mlflow_sktime.save_model(sktime_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match=f"Method {mlflow_sktime.SKTIME_PREDICT_PROBA} requires passing a "
        f"dictionary.",
    ):
        loaded_pyfunc.predict(pd.DataFrame())


@pytest.mark.skipif(
    not _check_soft_dependencies("mlflow", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_pyfunc_predict_proba_raises_invalid_dict_value(auto_arima_model, model_path):
    """Test pyfunc predict_proba raises exception with invalid dict value."""
    from mlflow.exceptions import MlflowException

    from sktime.utils import mlflow_sktime

    auto_arima_model.pyfunc_predict_conf = {
        "predict_method": {"predict_proba": {"marginal": True}}
    }
    mlflow_sktime.save_model(sktime_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow_sktime.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(
        MlflowException,
        match=f"Method {mlflow_sktime.SKTIME_PREDICT_PROBA} requires passing "
        f"quantile values.",
    ):
        loaded_pyfunc.predict(pd.DataFrame())
