"""Collect ``python_dependencies`` for an isolated benchmark pair."""

from inspect import isclass

from sktime.base import BaseEstimator
from sktime.benchmarking._benchmarking_dataclasses import TaskObject


def _normalize_python_dependencies(deps) -> list[str]:
    """Coerce a ``python_dependencies`` tag value to a flat list of strings."""
    if deps is None:
        return []
    if isinstance(deps, str):
        return [deps] if deps else []
    if isinstance(deps, list):
        result = []
        for dep in deps:
            if isinstance(dep, str) and dep:
                result.append(dep)
            elif isinstance(dep, list) and dep:
                # disjunction: pick the first alternative, same as registry.deps
                first = dep[0]
                if isinstance(first, str) and first:
                    result.append(first)
        return result
    return []


def _object_python_dependencies(obj) -> list[str]:
    """Collect ``python_dependencies`` from a sktime object, class, or dataset."""
    if obj is None:
        return []

    if isclass(obj) and issubclass(obj, BaseEstimator):
        return _normalize_python_dependencies(obj.get_class_tag("python_dependencies"))

    if hasattr(obj, "get_class_tag"):
        return _normalize_python_dependencies(obj.get_class_tag("python_dependencies"))

    return []


def _dataset_loader_dependencies(dataset_loader) -> list[str]:
    """Collect dependencies required by a dataset loader."""
    if dataset_loader is None:
        return []

    if isclass(dataset_loader) or hasattr(dataset_loader, "get_class_tag"):
        return _object_python_dependencies(dataset_loader)

    return []


def collect_pair_requirements(estimator: BaseEstimator, task: TaskObject) -> list[str]:
    """Collect PEP 440 dependency strings for a task-estimator pair.

    Dependencies are derived from ``python_dependencies`` tags on the
    estimator and task components (scorers, splitters, dataset loaders).
    """
    requirements: list[str] = []
    requirements.extend(_object_python_dependencies(estimator))

    for scorer in task.scorers:
        requirements.extend(_object_python_dependencies(scorer))

    for component in (
        task.cv_splitter,
        task.cv_global,
        task.cv_global_temporal,
        task.cv_X,
    ):
        requirements.extend(_object_python_dependencies(component))

    requirements.extend(_dataset_loader_dependencies(task.data))

    return sorted(set(requirements))
