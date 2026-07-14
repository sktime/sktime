# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test that type-specific modules do not cross-import each other at module level.

Three-tier import hierarchy:
- base framework (base, datatypes, registry, utils, etc.) — anyone may import
- networks — may import base framework only; type modules may import networks
- type modules (forecasting, transformations, classification, etc.) — may import
  networks or base framework, but must not cross-import each other

Imports inside functions or classes are allowed (deferred, not module-level).
Imports inside TYPE_CHECKING blocks are also checked — prefer lazy imports instead.

Existing violations are listed in KNOWN_EXCEPTIONS and should be cleaned up
in follow-up PRs. New violations will fail this test.
"""

__author__ = ["Nischal1425"]

import ast
from pathlib import Path

import pytest

# Type-specific modules that must not cross-import each other.
TYPE_MODULES = frozenset(
    {
        "alignment",
        "classification",
        "clustering",
        "detection",
        "forecasting",
        "param_est",
        "regression",
        "transformations",
    }
)

# networks sits between base framework and type modules:
# type modules may import networks, but networks must not import any type module.
_NETWORKS = "networks"
_CHECKED_SOURCES = TYPE_MODULES | {_NETWORKS}

# Existing violations: (path relative to sktime root, source module, imported module).
# These are pre-existing and should be fixed in follow-up PRs, not in this one.
KNOWN_EXCEPTIONS = frozenset(
    {
        (
            "classification/dictionary_based/_boss.py",
            "classification",
            "transformations",
        ),
        (
            "classification/dictionary_based/_muse.py",
            "classification",
            "transformations",
        ),
        (
            "classification/dictionary_based/_tde.py",
            "classification",
            "transformations",
        ),
        (
            "classification/dictionary_based/_weasel.py",
            "classification",
            "transformations",
        ),
        (
            "classification/distance_based/_elastic_ensemble.py",
            "classification",
            "transformations",
        ),
        (
            "classification/distance_based/_proximity_forest.py",
            "classification",
            "transformations",
        ),
        (
            "classification/distance_based/_shape_dtw.py",
            "classification",
            "transformations",
        ),
        (
            "classification/distance_based/tests/test_time_series_neighbors.py",
            "classification",
            "alignment",
        ),
        ("classification/ensemble/_ctsf.py", "classification", "transformations"),
        (
            "classification/ensemble/tests/test_ensemble.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_catch22_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_fresh_prince.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_matrix_profile_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_random_interval_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_signature_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_summary_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_tsfresh_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/interval_based/_cif.py",
            "classification",
            "transformations",
        ),
        (
            "classification/interval_based/_drcif.py",
            "classification",
            "transformations",
        ),
        (
            "classification/kernel_based/_arsenal.py",
            "classification",
            "transformations",
        ),
        (
            "classification/kernel_based/_rocket_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/plotting/temporal_importance_diagram.py",
            "classification",
            "transformations",
        ),
        (
            "classification/shapelet_based/_stc.py",
            "classification",
            "transformations",
        ),
        (
            "classification/tests/_classification_test_reproduction.py",
            "classification",
            "transformations",
        ),
        (
            "classification/tests/test_sklearn_compatibility.py",
            "classification",
            "transformations",
        ),
        ("clustering/compose/_as_transform.py", "clustering", "transformations"),
        ("clustering/compose/_pipeline.py", "clustering", "transformations"),
        (
            "clustering/compose/tests/test_pipeline.py",
            "clustering",
            "transformations",
        ),
        ("detection/compose/_as_transform.py", "detection", "transformations"),
        ("detection/eagglo.py", "detection", "transformations"),
        ("detection/stray.py", "detection", "transformations"),
        (
            "forecasting/base/tests/test_base_bugs.py",
            "forecasting",
            "transformations",
        ),
        (
            "forecasting/compose/tests/test_bagging.py",
            "forecasting",
            "transformations",
        ),
        (
            "forecasting/compose/tests/test_column_ensemble.py",
            "forecasting",
            "transformations",
        ),
        (
            "forecasting/compose/tests/test_groupbycategoryforecaster.py",
            "forecasting",
            "transformations",
        ),
        (
            "forecasting/compose/tests/test_hierarchy_ensemble.py",
            "forecasting",
            "transformations",
        ),
        (
            "forecasting/compose/tests/test_pipeline.py",
            "forecasting",
            "transformations",
        ),
        ("forecasting/compose/tests/test_reduce.py", "forecasting", "regression"),
        (
            "forecasting/compose/tests/test_reduce.py",
            "forecasting",
            "transformations",
        ),
        (
            "forecasting/compose/tests/test_reduce_global.py",
            "forecasting",
            "transformations",
        ),
        (
            "forecasting/compose/tests/test_transformer_select_forecaster.py",
            "forecasting",
            "transformations",
        ),
        (
            "forecasting/model_selection/tests/test_tune.py",
            "forecasting",
            "transformations",
        ),
        ("forecasting/tests/test_reconcile.py", "forecasting", "transformations"),
        ("param_est/compose/_pipeline.py", "param_est", "transformations"),
        ("param_est/plugin/_forecaster.py", "param_est", "forecasting"),
        ("param_est/plugin/_transformer.py", "param_est", "transformations"),
        ("param_est/tests/test_plugin.py", "param_est", "forecasting"),
        ("param_est/tests/test_plugin.py", "param_est", "transformations"),
        (
            "regression/tests/test_categorical_in_composite.py",
            "regression",
            "transformations",
        ),
        (
            "transformations/detrend/tests/test_detrend.py",
            "transformations",
            "forecasting",
        ),
        (
            "transformations/summarize/tests/test_FittedParamExtractor.py",
            "transformations",
            "forecasting",
        ),
        (
            "transformations/tests/test_imputer.py",
            "transformations",
            "forecasting",
        ),
        (
            "transformations/tests/test_multiplexer.py",
            "transformations",
            "forecasting",
        ),
        ("transformations/tests/test_subset.py", "transformations", "forecasting"),
        (
            "transformations/tests/test_transformif.py",
            "transformations",
            "param_est",
        ),
    }
)

_SKTIME_ROOT = Path(__file__).parent.parent


def _is_type_checking_block(node):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _get_import_target(node):
    """Return the top-level sktime submodule imported, or None."""
    if isinstance(node, ast.ImportFrom):
        if node.module and node.module.startswith("sktime."):
            parts = node.module.split(".")
            if len(parts) >= 2:
                return parts[1]
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith("sktime."):
                parts = alias.name.split(".")
                if len(parts) >= 2:
                    return parts[1]
    return None


def _collect_violations():
    """Return set of (rel_path, source_module, imported_module) for cross-imports."""
    violations = set()
    for py_file in _SKTIME_ROOT.rglob("*.py"):
        rel = py_file.relative_to(_SKTIME_ROOT)
        parts = rel.parts
        if not parts or parts[0] not in _CHECKED_SOURCES:
            continue
        source = parts[0]
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.iter_child_nodes(tree):
            # gather import nodes: direct top-level ones, plus those inside
            # TYPE_CHECKING blocks (which must also not cross-import type modules)
            imp_nodes = []
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imp_nodes.append(node)
            elif _is_type_checking_block(node):
                for inner in ast.iter_child_nodes(node):
                    if isinstance(inner, (ast.Import, ast.ImportFrom)):
                        imp_nodes.append(inner)
            for imp_node in imp_nodes:
                target = _get_import_target(imp_node)
                # violation: target is a type module and differs from source
                # type_module -> type_module: caught
                # networks    -> type_module: caught (networks not in TYPE_MODULES)
                # type_module -> networks:    allowed (networks not in TYPE_MODULES)
                if target and target != source and target in TYPE_MODULES:
                    violations.add((str(rel).replace("\\", "/"), source, target))
    return violations


def test_no_new_cross_module_imports():
    """No new module-level cross-imports between type-specific sktime modules.

    Type-specific modules (forecasting, transformations, classification, etc.)
    must not import each other at module level to avoid cascading import chains
    that can lead to circular imports. Networks may be imported by type modules
    but must not itself import any type module.

    Existing violations are grandfathered in KNOWN_EXCEPTIONS. Only new ones fail.
    To fix an exception: move the import inside the function/class that needs it,
    then remove its entry from KNOWN_EXCEPTIONS.
    """
    violations = _collect_violations()
    new_violations = violations - KNOWN_EXCEPTIONS
    stale_exceptions = KNOWN_EXCEPTIONS - violations

    messages = []
    if new_violations:
        lines = "\n".join(
            f"  {path}  ({src} -> {tgt})" for path, src, tgt in sorted(new_violations)
        )
        messages.append(
            f"New cross-module imports found (move import inside the function "
            f"or class that uses it):\n{lines}"
        )
    if stale_exceptions:
        lines = "\n".join(
            f"  {path}  ({src} -> {tgt})" for path, src, tgt in sorted(stale_exceptions)
        )
        messages.append(
            f"KNOWN_EXCEPTIONS entries no longer violated (remove them):\n{lines}"
        )

    if messages:
        pytest.fail("\n\n".join(messages))
