#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""The ``mlflow_sktime`` module provides an MLflow API for ``sktime`` models.

This module exports ``sktime`` models in the following formats:

sktime (native) format
    This is the main flavor that can be loaded back into sktime, which relies on pickle
    internally to serialize a model.
mlflow.pyfunc
    Produced for use by generic pyfunc-based deployment tools and batch inference.

    The ``pyfunc`` flavor of the model supports sktime predict methods ``predict``,
    ``predict_interval``, ``predict_quantiles`` and ``predict_var`` (``predict_proba``
    and ``predict_residuals`` are currently not supported).
"""

__author__ = ["benjaminbluhm"]
__all__ = [
    "get_default_pip_requirements",
    "get_default_conda_env",
    "save_model",
    "log_model",
    "load_model",
]

import logging
import os
import pickle

import pandas as pd
import yaml

import sktime
from sktime import utils
from sktime.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies("mlflow", severity="warning"):
    from mlflow import pyfunc

FLAVOR_NAME = "mlflow_sktime"
_MODEL_BINARY_KEY = "data"
_MODEL_BINARY_FILE_NAME = "model.skt"
_MODEL_TYPE_KEY = "model_type"
PREDICT_METHODS = ["predict", "predict_interval", "predict_quantiles", "predict_var"]
SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"
SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE,
]

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(include_cloudpickle=False):
    """Create list of default pip requirements for MLflow Models.

    Returns
    -------
    list of default pip requirements for MLflow Models produced by this flavor.
    Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
    that, at a minimum, contains these requirements.
    """
    _check_soft_dependencies("mlflow", severity="error")
    from mlflow.utils.requirements_utils import _get_pinned_requirement

    pip_deps = [_get_pinned_requirement("sktime")]
    if include_cloudpickle:
        pip_deps += [_get_pinned_requirement("cloudpickle")]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """Return default Conda environment for MLflow Models.

    Returns
    -------
    The default Conda environment for MLflow Models produced by calls to
    :func:`save_model()` and :func:`log_model()`
    """
    _check_soft_dependencies("mlflow", severity="error")
    from mlflow.utils.environment import _mlflow_conda_env

    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements(include_cloudpickle)
    )


def save_model(
    sktime_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    serialization_format=SERIALIZATION_FORMAT_PICKLE,
):  # TODO: can we specify a type for fitted instance of sktime model below?
    """Save a sktime model to a path on the local file system.

    Parameters
    ----------
    sktime_model :
        Fitted sktime model object.
    path : str
        Local path where the model is to be saved.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    mlflow_model: mlflow.models.Model, optional (default=None)
        mlflow.models.Model configuration to which to add the python_function flavor.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models.signature import infer_signature
          train = df.drop_column("target_label")
          predictions = ... # compute model predictions
          signature = infer_signature(train, predictions)

        .. Warning:: if performing probabilistic forecasts (``predict_interval``,
          ``predict_quantiles``) with a sktime model, the signature
          on the returned prediction object will not be correctly inferred due
          to the Pandas MultiIndex column type when using the these methods.
          ``infer_schema`` will function correctly if using the ``pyfunc`` flavor
          of the model, though.
    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["sktime", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"

    References
    ----------
    .. [1] https://www.mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.Model.save

    Examples
    --------
    >>> from sktime.datasets import load_airline  # doctest: +SKIP
    >>> from sktime.forecasting.arima import ARIMA  # doctest: +SKIP
    >>> from sktime.utils import mlflow_sktime  # doctest: +SKIP
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = ARIMA(  # doctest: +SKIP
    ...     order=(1, 1, 0),
    ...     seasonal_order=(0, 1, 0, 12),
    ...     suppress_warnings=True)
    >>> forecaster.fit(y)  # doctest: +SKIP
    ARIMA(...)
    >>> model_path = "model"  # doctest: +SKIP
    >>> mlflow_sktime.save_model(  # doctest: +SKIP
    ...     sktime_model=forecaster,
    ...     path=model_path)  # doctest: +SKIP
    >>> loaded_model = mlflow_sktime.load_model(model_uri=model_path)  # doctest: +SKIP
    >>> loaded_model.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """  # noqa: E501
    _check_soft_dependencies("mlflow", severity="error")
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.models import Model
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.models.utils import _save_example
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
    from mlflow.utils.environment import (
        _CONDA_ENV_FILE_NAME,
        _CONSTRAINTS_FILE_NAME,
        _PYTHON_ENV_FILE_NAME,
        _REQUIREMENTS_FILE_NAME,
        _process_conda_env,
        _process_pip_requirements,
        _PythonEnv,
        _validate_env_arguments,
    )
    from mlflow.utils.file_utils import write_to
    from mlflow.utils.model_utils import (
        _validate_and_copy_code_paths,
        _validate_and_prepare_target_save_path,
    )

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. "
                "Please specify one of the following supported formats: "
                "{supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_path = os.path.join(path, _MODEL_BINARY_FILE_NAME)
    _save_model(
        sktime_model, model_data_path, serialization_format=serialization_format
    )

    model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="sktime.utils.mlflow_sktime",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_bin_kwargs,
    )
    flavor_conf = {
        _MODEL_TYPE_KEY: sktime_model.__class__.__name__,
        **model_bin_kwargs,
    }
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        sktime_version=sktime.__version__,
        code=code_dir_subpath,
        serialization_format=serialization_format,
        **flavor_conf,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            include_cloudpickle = (
                serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
            )
            default_reqs = get_default_pip_requirements(include_cloudpickle)
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def log_model(
    sktime_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature=None,
    input_example=None,
    await_registration_for=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    serialization_format=SERIALIZATION_FORMAT_PICKLE,
    **kwargs,
):  # TODO: can we specify a type for fitted instance of sktime model below?
    """
    Log a sktime model as an MLflow artifact for the current run.

    Parameters
    ----------
    sktime_model : fitted sktime model
        Fitted sktime model object.
    artifact_path : str
        Run-relative artifact path to save the model to.
    conda_env : Union[dict, str], optional (default=None)
        Either a dictionary representation of a Conda environment or the path to a
        conda environment yaml file.
    code_paths : array-like, optional (default=None)
        A list of local filesystem paths to Python file dependencies (or directories
        containing file dependencies). These files are *prepended* to the system path
        when the model is loaded.
    registered_model_name : str, optional (default=None)
        If given, create a model version under ``registered_model_name``, also creating
        a registered model if one with the given name does not exist.
    signature : mlflow.models.signature.ModelSignature, optional (default=None)
        Model Signature mlflow.models.ModelSignature describes
        model input and output :py:class:`Schema <mlflow.types.Schema>`. The model
        signature can be :py:func:`inferred <mlflow.models.infer_signature>` from
        datasets with valid model input (e.g. the training dataset with target column
        omitted) and valid model output (e.g. model predictions generated on the
        training dataset), for example:

        .. code-block:: py

          from mlflow.models.signature import infer_signature
          train = df.drop_column("target_label")
          predictions = ... # compute model predictions
          signature = infer_signature(train, predictions)

        .. Warning:: if performing probabilistic forecasts (``predict_interval``,
          ``predict_quantiles``) with a sktime model, the signature
          on the returned prediction object will not be correctly inferred due
          to the Pandas MultiIndex column type when using the these methods.
          ``infer_schema`` will function correctly if using the ``pyfunc`` flavor
          of the model, though. The ``pyfunc`` flavor of the model supports sktime
          predict methods ``predict``, ``predict_interval``, ``predict_quantiles``
          and ``predict_var`` while ``predict_proba`` and ``predict_residuals`` are
          currently not supported.
    input_example : Union[pandas.core.frame.DataFrame, numpy.ndarray, dict, list, csr_matrix, csc_matrix], optional (default=None)
        Input example provides one or several instances of valid model input.
        The example can be used as a hint of what data to feed the model. The given
        example will be converted to a ``Pandas DataFrame`` and then serialized to json
        using the ``Pandas`` split-oriented format. Bytes are base64-encoded.
    await_registration_for : int, optional (default=None)
        Number of seconds to wait for the model version to finish being created and is
        in ``READY`` status. By default, the function waits for five minutes. Specify 0
        or None to skip waiting.
    pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["sktime", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    extra_pip_requirements : Union[Iterable, str], optional (default=None)
        Either an iterable of pip requirement strings
        (e.g. ["pandas", "-r requirements.txt", "-c constraints.txt"]) or the string
        path to a pip requirements file on the local filesystem
        (e.g. "requirements.txt")
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"
    kwargs:
        Additional arguments for :py:class:`mlflow.models.model.Model`

    Returns
    -------
    A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
    metadata of the logged model.

    See Also
    --------
    MLflow

    References
    ----------
    .. [1] https://www.mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.Model.log

    >>> import mlflow  # doctest: +SKIP
    >>> from mlflow.utils.environment import _mlflow_conda_env  # doctest: +SKIP
    >>> from sktime.datasets import load_airline  # doctest: +SKIP
    >>> from sktime.forecasting.arima import ARIMA  # doctest: +SKIP
    >>> from sktime.utils import mlflow_sktime  # doctest: +SKIP
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = ARIMA(  # doctest: +SKIP
    ...     order=(1, 1, 0),
    ...     seasonal_order=(0, 1, 0, 12),
    ...     suppress_warnings=True)
    >>> forecaster.fit(y)  # doctest: +SKIP
    ARIMA(...)
    >>> mlflow.start_run()  # doctest: +SKIP
    >>> artifact_path = "model"  # doctest: +SKIP
    >>> model_info = mlflow_sktime.log_model(
    ...     sktime_model=forecaster,
    ...     artifact_path=artifact_path)  # doctest: +SKIP
    """  # noqa: E501
    _check_soft_dependencies("mlflow", severity="error")
    from mlflow.models import Model

    if await_registration_for is None:
        from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

        await_registration_for = DEFAULT_AWAIT_MAX_SLEEP_SECONDS

    return Model.log(
        artifact_path=artifact_path,
        flavor=utils.mlflow_sktime,
        registered_model_name=registered_model_name,
        sktime_model=sktime_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        serialization_format=serialization_format,
        **kwargs,
    )


def load_model(
    model_uri, dst_path=None, serialization_format=SERIALIZATION_FORMAT_PICKLE
):
    """
    Load a sktime model from a local file or a run.

    Parameters
    ----------
    model_uri : str
        The location, in URI format, of the MLflow model. For example:

                    - ``/Users/me/path/to/local/model``
                    - ``relative/path/to/local/model``
                    - ``s3://my_bucket/path/to/model``
                    - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                    - ``mlflow-artifacts:/path/to/model``

        For more information about supported URI schemes, see
        `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
        artifact-locations>`_.
    dst_path : str, optional (default=None)
        The local filesystem path to which to download the model artifact.This
        directory must already exist. If unspecified, a local output path will
        be created.
    serialization_format : str, optional (default="pickle")
        The format in which to serialize the model. This should be one of the formats
        "pickle" or "cloudpickle"

    Returns
    -------
    A sktime model instance.

    References
    ----------
    .. [1] https://www.mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.Model.load

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.arima import ARIMA
    >>> from sktime.utils import mlflow_sktime  # doctest: +SKIP
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = ARIMA(  # doctest: +SKIP
    ...     order=(1, 1, 0),
    ...     seasonal_order=(0, 1, 0, 12),
    ...     suppress_warnings=True)
    >>> forecaster.fit(y)  # doctest: +SKIP
    ARIMA(...)
    >>> model_path = "model"  # doctest: +SKIP
    >>> mlflow_sktime.save_model(  # doctest: +SKIP
    ...     sktime_model=forecaster,
    ...     path=model_path)
    >>> loaded_model = mlflow_sktime.load_model(model_uri=model_path)  # doctest: +SKIP
    """  # noqa: E501
    _check_soft_dependencies("mlflow", severity="error")
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
    from mlflow.utils.model_utils import (
        _add_code_from_conf_to_system_path,
        _get_flavor_configuration,
    )

    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    sktime_model_file_path = os.path.join(
        local_model_path, flavor_conf.get(_MODEL_BINARY_KEY, _MODEL_BINARY_FILE_NAME)
    )

    return _load_model(
        sktime_model_file_path, serialization_format=serialization_format
    )


def _save_model(model, path, serialization_format=SERIALIZATION_FORMAT_PICKLE):

    _check_soft_dependencies("mlflow", severity="error")
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

    with open(path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            _check_soft_dependencies("cloudpickle", severity="error")
            import cloudpickle

            cloudpickle.dump(model, out)
        else:
            raise MlflowException(
                message="Unrecognized serialization format: "
                "{serialization_format}".format(
                    serialization_format=serialization_format
                ),
                error_code=INTERNAL_ERROR,
            )


def _load_model(path, serialization_format=SERIALIZATION_FORMAT_PICKLE):

    _check_soft_dependencies("mlflow", severity="error")
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. "
                "Please specify one of the following supported formats: "
                "{supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    with open(path, "rb") as pickled_model:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(pickled_model)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            _check_soft_dependencies("cloudpickle", severity="error")
            import cloudpickle

            return cloudpickle.load(pickled_model)


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Parameters
    ----------
    path : str
        Local filesystem path to the MLflow Model with the sktime flavor.

    See Also
    --------
    MLflow

    References
    ----------
    .. [1] https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model
    """  # noqa: E501
    return _SktimeModelWrapper(_load_model(path))


class _SktimeModelWrapper:
    def __init__(self, sktime_model):
        _check_soft_dependencies("mlflow", severity="error")
        self.sktime_model = sktime_model

    def predict(self, dataframe) -> pd.DataFrame:

        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

        if len(dataframe) > 1:
            raise MlflowException(
                f"The provided prediction pd.DataFrame contains {len(dataframe)} rows. "
                "Only 1 row should be supplied.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        attrs = dataframe.to_dict(orient="index").get(0)
        fh = attrs.get("fh", None)
        X = attrs.get("X", None)

        predict_method = attrs.get("predict_method", "predict")

        if predict_method not in PREDICT_METHODS:
            raise MlflowException(
                f"The provided `predict_method` value {predict_method} "
                f"must be one of {PREDICT_METHODS}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if predict_method == "predict":
            y_pred = self.sktime_model.predict(fh=fh, X=X)

        if predict_method == "predict_interval":
            coverage = attrs.get("coverage", 0.9)
            y_pred = self.sktime_model.predict_interval(fh=fh, X=X, coverage=coverage)
            # Signature inference does not support pandas MultiIndex column format
            y_pred.columns = y_pred.columns.to_flat_index().map(
                lambda x: "".join(str(x))
            )

        if predict_method == "predict_quantiles":
            alpha = attrs.get("alpha", None)
            y_pred = self.sktime_model.predict_quantiles(fh=fh, X=X, alpha=alpha)
            # Signature inference does not support pandas MultiIndex column format
            y_pred.columns = y_pred.columns.to_flat_index().map(
                lambda x: "".join(str(x))
            )

        if predict_method == "predict_var":
            cov = attrs.get("cov", False)
            y_pred = self.sktime_model.predict_var(fh=fh, X=X, cov=cov)

        return y_pred


# TODO: Add support for autologging
