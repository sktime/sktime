"""Dependency checking utility functionality."""

from sktime.utils.dependencies._dependencies import (
    _check_dl_dependencies,
    _check_env_marker,
    _check_estimator_deps,
    _check_mlflow_dependencies,
    _check_python_version,
    _check_soft_dependencies,
)
from sktime.utils.dependencies._isinstance import _isinstance_by_name
from sktime.utils.dependencies._placeholder import _placeholder_record
from sktime.utils.dependencies._safe_import import _safe_import

__all__ = [
    "_check_dl_dependencies",
    "_check_env_marker",
    "_check_estimator_deps",
    "_check_mlflow_dependencies",
    "_check_python_version",
    "_check_soft_dependencies",
    "_isinstance_by_name",
    "_placeholder_record",
    "_safe_import",
]
