"""Collect ``python_dependencies`` for an isolated benchmark pair."""

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
    """Collect ``python_dependencies`` via ``get_class_tag`` if present."""
    if obj is None or not hasattr(obj, "get_class_tag"):
        return []
    return _normalize_python_dependencies(obj.get_class_tag("python_dependencies"))


def collect_pair_requirements(estimator: BaseEstimator, task: TaskObject) -> list[str]:
    """Collect PEP 440 dependency strings for a task-estimator pair.

    Dependencies are derived from ``python_dependencies`` tags on the
    estimator and task components (scorers, splitters, dataset classes).
    """
    components = (
        estimator,
        task.data,
        *task.scorers,
        task.cv_splitter,
        task.cv_global,
        task.cv_global_temporal,
        task.cv_X,
    )
    requirements = [
        dep
        for component in components
        for dep in _object_python_dependencies(component)
    ]
    return sorted(set(requirements))
