"""Dependency checking utility functionality."""

from sktime.utils.dependencies._dependencies import (
    _check_dl_dependencies,
    _check_env_marker,
    _check_estimator_deps,
    _check_mlflow_dependencies,
    _check_python_version,
    _check_soft_dependencies,
)
from sktime.utils.dependencies._placeholder import _placeholder_record

__all__ = [
    "_check_dl_dependencies",
    "_check_env_marker",
    "_check_estimator_deps",
    "_check_mlflow_dependencies",
    "_check_python_version",
    "_check_soft_dependencies",
    "_placeholder_record",
]
